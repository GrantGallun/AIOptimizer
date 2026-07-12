"""Public, lazy local-encoder helpers shared by product and research code."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Sequence


DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
EmbedFn = Callable[[list[str]], Any]


@lru_cache(maxsize=2)
def _load_encoder(model_name: str) -> tuple[Any, Any, Any]:
    """Load a local encoder only when embedding is requested."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True)
    model.eval()
    return torch, tokenizer, model


def embed_texts(texts: list[str], model_name: str = DEFAULT_ENCODER) -> Any:
    """Return normalized mean-pooled embeddings for ``texts``."""
    torch, tokenizer, model = _load_encoder(model_name)
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    summed = (output.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = summed / counts
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def as_vector(value: Any) -> list[float]:
    """Convert torch, numpy, or plain vectors to a common representation."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, with zero for either zero-length vector."""
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


__all__ = ["DEFAULT_ENCODER", "EmbedFn", "as_vector", "cosine", "embed_texts"]
