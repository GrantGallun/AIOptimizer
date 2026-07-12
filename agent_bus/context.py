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
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from aioptimizer.encoder import DEFAULT_ENCODER, as_vector, cosine, embed_texts

Chunk = dict[str, Any]
EmbedFn = Callable[[list[str]], Any]
AbstractFn = Callable[[Any], Any]

# Backward-compatible private aliases used by frozen research modules.
_encoder_embed = embed_texts
_as_vector = as_vector
_cosine = cosine


from aioptimizer.selection import ContextCompactor  # moved into the product package


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
