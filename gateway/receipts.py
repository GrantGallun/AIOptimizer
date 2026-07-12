"""Shadow-A/B receipts: the evidence that an optimization preserved quality.

For a deterministic sample of optimized requests, the gateway ALSO sends the
original (unoptimized) body upstream and judges the two responses with the same
encoder used across the research (all-MiniLM). Every judged pair lands in the
ledger; ``gateway.report`` turns the ledger into savings + a quality-parity rate
with a Wilson interval. Sampling is content-hash based, so a replayed ledger is
reproducible — determinism at the decision boundary, per FINDINGS.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable

EmbedFn = Callable[[list[str]], Any]
DEFAULT_PARITY_THRESHOLD = 0.90


def _default_embed(texts: list[str]) -> Any:
    # Same local MiniLM encoder the research stack uses (lazy import: torch/transformers
    # only load if a judge actually runs without an injected embed_fn).
    from agent_bus.context import DEFAULT_ENCODER, _encoder_embed

    return _encoder_embed(texts, model_name=DEFAULT_ENCODER)


def _as_vector(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def response_text(response: dict[str, Any]) -> str:
    """Extract assistant text from OpenAI, Anthropic, or Ollama response shapes."""
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError):
        pass
    content = response.get("content")
    if isinstance(content, list):
        texts = [
            block.get("text")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if texts:
            return "\n".join(texts)
    value = response.get("response")
    return str(value) if isinstance(value, str) else ""


class ShadowJudge:
    """Judge raw-vs-optimized responses; deterministic content-hash sampling."""

    def __init__(
        self,
        *,
        rate: float = 0.1,
        parity_threshold: float = DEFAULT_PARITY_THRESHOLD,
        embed_fn: EmbedFn | None = None,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("rate must be between 0 and 1")
        self.rate = rate
        self.parity_threshold = parity_threshold
        self._embed_fn = embed_fn

    def should_sample(self, body: dict[str, Any]) -> bool:
        """Deterministic per-content decision so a replayed stream samples identically."""
        if self.rate <= 0.0:
            return False
        if self.rate >= 1.0:
            return True
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF < self.rate

    def judge(self, raw_text: str, optimized_text: str) -> dict[str, Any]:
        if raw_text == optimized_text:
            similarity = 1.0
        else:
            embed = self._embed_fn or _default_embed
            vectors = [_as_vector(v) for v in embed([raw_text, optimized_text])]
            similarity = _cosine(vectors[0], vectors[1])
        return {
            "similarity": round(similarity, 4),
            "parity": similarity >= self.parity_threshold,
        }
