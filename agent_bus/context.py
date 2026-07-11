#!/usr/bin/env python3
"""Reusable encoder-ranked context selection and compaction.

The public API deliberately works with small structured chunks rather than raw
strings so callers can retain provenance while changing the amount of context
sent to a model.  The default embedder is the same local MiniLM encoder used by
the brain-runtime context-selection experiment; tests and applications may
inject an ``embed_fn`` to avoid a model dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
Chunk = dict[str, Any]
EmbedFn = Callable[[list[str]], Any]
AbstractFn = Callable[[Any], Any]


@lru_cache(maxsize=2)
def _load_encoder(model_name: str) -> tuple[Any, Any, Any]:
    """Load the local attention encoder only when the default embedder is used."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()
    return torch, tokenizer, model


def _encoder_embed(texts: list[str], *, model_name: str) -> Any:
    torch, tokenizer, model = _load_encoder(model_name)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    summed = (output.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = summed / counts
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def _as_vector(value: Any) -> list[float]:
    """Convert torch/numpy/plain vectors to a small common representation."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class ContextCompactor:
    """Select relevant chunks and compact the remainder into abstractions.

    ``chunks`` are mappings with ``id`` and ``text`` keys.  ``select`` returns
    shallow-copied mappings in descending encoder-cosine order.  ``compact``
    returns full selected chunks first, followed by abstract chunks, with the
    sum of returned text lengths no greater than ``budget_chars``.  An
    ``abstract_fn`` normally receives a chunk's text; for compatibility with
    structured callbacks, a callback that rejects text is retried with the
    complete chunk mapping.
    """

    def __init__(self, embed_fn: EmbedFn | None = None, *, model_name: str = DEFAULT_ENCODER) -> None:
        self.model_name = model_name
        self._embed_fn = embed_fn

    def _embed(self, texts: list[str]) -> Any:
        if self._embed_fn is not None:
            return self._embed_fn(texts)
        return _encoder_embed(texts, model_name=self.model_name)

    @staticmethod
    def _validate_chunks(chunks: Sequence[Mapping[str, Any]]) -> list[Chunk]:
        normalized: list[Chunk] = []
        for chunk in chunks:
            if not isinstance(chunk, Mapping) or "id" not in chunk or "text" not in chunk:
                raise ValueError("each chunk must be a mapping with id and text")
            if not isinstance(chunk["text"], str):
                raise ValueError("chunk text must be a string")
            normalized.append(dict(chunk))
        return normalized

    def _rank(self, query: str, chunks: Sequence[Mapping[str, Any]]) -> list[Chunk]:
        normalized = self._validate_chunks(chunks)
        if not normalized:
            return []
        vectors = self._embed([query] + [chunk["text"] for chunk in normalized])
        vectors = [_as_vector(vector) for vector in vectors]
        if len(vectors) != len(normalized) + 1:
            raise ValueError("embed_fn must return one vector per input text")
        scores = [_cosine(vectors[0], vector) for vector in vectors[1:]]
        order = sorted(range(len(normalized)), key=lambda index: (-scores[index], index))
        return [normalized[index] for index in order]

    def select(self, query: str, chunks: Sequence[Mapping[str, Any]], k: int) -> list[Chunk]:
        """Return the top ``k`` chunks ranked by encoder cosine similarity."""
        if k < 0:
            raise ValueError("k must be non-negative")
        normalized = self._validate_chunks(chunks)
        if k == 0 or not normalized:
            return []
        return self._rank(query, normalized)[:k]

    @staticmethod
    def _abstract(abstract_fn: AbstractFn, chunk: Chunk) -> str:
        try:
            value = abstract_fn(chunk["text"])
        except (KeyError, TypeError):
            value = abstract_fn(chunk)
        if isinstance(value, Mapping):
            value = value.get("text", "")
        return str(value)

    def compact(
        self,
        query: str,
        chunks: Sequence[Mapping[str, Any]],
        budget_chars: int,
        abstract_fn: AbstractFn | None = None,
    ) -> list[Chunk]:
        """Keep the most relevant full chunks and optionally abstract the rest."""
        if budget_chars < 0:
            raise ValueError("budget_chars must be non-negative")
        normalized = self._validate_chunks(chunks)
        if not normalized or budget_chars == 0:
            return []

        ranked = self._rank(query, normalized)
        full: list[Chunk] = []
        remainder: list[Chunk] = []
        used = 0
        for chunk in ranked:
            size = len(chunk["text"])
            if used + size <= budget_chars:
                full.append(chunk)
                used += size
            else:
                remainder.append(chunk)

        if abstract_fn is None:
            return full

        compacted = list(full)
        for chunk in remainder:
            remaining = budget_chars - used
            if remaining <= 0:
                break
            text = self._abstract(abstract_fn, chunk)
            if not text:
                continue
            text = text[:remaining]
            compacted.append({"id": chunk["id"], "text": text, "abstract": True})
            used += len(text)
        return compacted


def _load_chunks(value: str) -> list[Chunk]:
    path = Path(value)
    if path.exists():
        value = path.read_text(encoding="utf-8")
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        raise ValueError("chunks JSON must be a list")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--chunks-json", required=True, help="JSON list, or a path containing one")
    parser.add_argument("--k", type=int, default=None, help="Select this many chunks")
    parser.add_argument("--budget-chars", type=int, default=None, help="Compact to this text budget")
    args = parser.parse_args()
    compactor = ContextCompactor()
    chunks = _load_chunks(args.chunks_json)
    if args.budget_chars is not None:
        result = compactor.compact(args.query, chunks, args.budget_chars)
    else:
        result = compactor.select(args.query, chunks, len(chunks) if args.k is None else args.k)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
