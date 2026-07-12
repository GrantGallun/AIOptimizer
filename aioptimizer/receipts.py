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
from typing import Any, Callable

from .encoder import as_vector, cosine, embed_texts

EmbedFn = Callable[[list[str]], Any]
DEFAULT_PARITY_THRESHOLD = 0.90


def _default_embed(texts: list[str]) -> Any:
    return embed_texts(texts)


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
            vectors = [as_vector(v) for v in embed([raw_text, optimized_text])]
            similarity = cosine(vectors[0], vectors[1])
        return {
            "similarity": round(similarity, 4),
            "parity": similarity >= self.parity_threshold,
        }
