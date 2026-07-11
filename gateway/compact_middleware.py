"""Context-compaction middleware: HYP-20/21 as a live gateway optimization.

Evidence (FINDINGS §2): as context grows, dumping everything degrades the model
("lost in the middle": 0.83 -> 0.33 on real code) while encoder top-k selection
holds at the oracle ceiling. This middleware applies that finding to oversized
requests: when a request exceeds ``budget_chars``, the longest USER message is
split into paragraph chunks and only the chunks most relevant to the latest user
message are kept (original order preserved). System messages are never touched.

Every compaction is visible to the receipts pipeline (the body changes, so the
shadow judge samples it) — savings never ship without quality evidence.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent_bus.context import ContextCompactor

DEFAULT_BUDGET_CHARS = 12_000
MIN_CHUNKS_TO_COMPACT = 4


class CompactContextMiddleware:
    """Keep the encoder-relevant slice of the longest user message when oversized."""

    def __init__(
        self,
        *,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        embed_fn: Callable[[list[str]], Any] | None = None,
    ) -> None:
        if budget_chars <= 0:
            raise ValueError("budget_chars must be positive")
        self.budget_chars = budget_chars
        self._compactor = ContextCompactor(embed_fn)

    def before_request(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return body
        total_chars = len(json.dumps(body, ensure_ascii=False))
        if total_chars <= self.budget_chars:
            return body

        # The compaction target: the longest user message (the context-stuffed one).
        user_indexes = [
            i
            for i, m in enumerate(messages)
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
        ]
        if not user_indexes:
            return body
        target = max(user_indexes, key=lambda i: len(messages[i]["content"]))
        content = messages[target]["content"]
        chunks = [
            {"id": i, "text": part.strip()}
            for i, part in enumerate(content.split("\n\n"))
            if part.strip()
        ]
        if len(chunks) < MIN_CHUNKS_TO_COMPACT:
            return body  # nothing meaningful to select between

        # Relevance anchor: the LATEST user message (the actual question).
        query = str(messages[user_indexes[-1]]["content"])[:300]
        excess = total_chars - self.budget_chars
        chunk_budget = max(500, len(content) - excess)
        kept = self._compactor.compact(query, chunks, chunk_budget)
        if not kept or len(kept) == len(chunks):
            return body
        # Restore original document order for prompt coherence.
        kept_sorted = sorted(kept, key=lambda c: c["id"])

        compacted = json.loads(json.dumps(body))  # deep copy; never mutate the original
        compacted["messages"][target]["content"] = "\n\n".join(c["text"] for c in kept_sorted)
        return compacted

    def after_response(self, body: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        return response
