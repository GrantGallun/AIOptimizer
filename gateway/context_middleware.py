"""Default-off attention-context middleware for oversized chat histories."""

from __future__ import annotations

import json
import threading
from typing import Any, Sequence

from gateway.context_compiler import DEFAULT_DENY_PATTERNS, ConversationCompiler


class AttentionContextMiddleware:
    """Replace prior chat turns with a locally organized, source-labelled working set.

    Original system messages and the active (latest) user message remain verbatim.
    The middleware only acts when the serialized request exceeds ``budget_chars``;
    gateway shadow receipts can therefore compare every sampled rewrite to raw.
    """

    def __init__(
        self,
        *,
        budget_chars: int = 12_000,
        compiler: ConversationCompiler | None = None,
        deny_patterns: Sequence[str] = DEFAULT_DENY_PATTERNS,
    ):
        if budget_chars <= 0:
            raise ValueError("budget_chars must be positive")
        self.budget_chars = budget_chars
        self.compiler = compiler or ConversationCompiler(deny_patterns=deny_patterns)
        self._local = threading.local()

    def _set_receipt(self, **values: Any) -> None:
        self._local.receipt = {"applied": False, **values}

    def receipt_metadata(self) -> dict[str, Any]:
        """Current handler-thread metadata consumed by the gateway ledger."""
        return dict(getattr(self._local, "receipt", {"applied": False}))

    def before_request(self, body: dict[str, Any]) -> dict[str, Any]:
        original_chars = len(json.dumps(body, ensure_ascii=False))
        self._set_receipt(original_chars=original_chars)
        messages = body.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            return body
        if len(json.dumps(body, ensure_ascii=False)) <= self.budget_chars:
            return body
        latest_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], dict)
                and messages[index].get("role") == "user"
                and isinstance(messages[index].get("content"), str)
            ),
            None,
        )
        if latest_user_index is None:
            return body
        latest_user = messages[latest_user_index]
        systems = [
            message
            for message in messages[:latest_user_index]
            if isinstance(message, dict)
            and message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and not self.compiler._is_denied(message["content"])
        ]
        history = [
            message
            for message in messages[:latest_user_index]
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant", "tool"}
            and isinstance(message.get("content"), str)
        ]
        if not history:
            return body

        compiled = self.compiler.compile(history)
        integrity = self.compiler.audit_integrity(compiled)
        if not integrity["ok"]:
            self._set_receipt(original_chars=original_chars, integrity_ok=False)
            return body
        cache_before = self.compiler.embedding_cache_stats()
        organized = self.compiler.organize(
            compiled,
            query=latest_user["content"],
            max_records=len(compiled.records),
        )
        candidates = [
            record
            for cluster in organized["clusters"]
            for record in cluster["records"]
        ]
        # ``organize`` pins the last historical user turn.  It is still history,
        # so include it in attention order rather than treating it as the active query.
        candidates.extend(organized["pinned"])
        fixed_chars = len(json.dumps({"messages": systems + [latest_user]}, ensure_ascii=False))
        header = "Relevant prior conversation (source-grounded):\n"
        available = self.budget_chars - fixed_chars - len(header) - 64
        if available <= 0:
            return body
        rendered = self.compiler._fit_sections([], candidates, available)
        if not rendered:
            return body
        rewritten = json.loads(json.dumps(body))
        rewritten["messages"] = systems + [
            {"role": "system", "content": header + rendered},
            dict(latest_user),
        ]
        cache_after = self.compiler.embedding_cache_stats()
        self._local.receipt = {
            "applied": True,
            "original_chars": original_chars,
            "rewritten_chars": len(json.dumps(rewritten, ensure_ascii=False)),
            "compiler_fingerprint": integrity["fingerprint"],
            "integrity_ok": True,
            "active_request_preserved": rewritten["messages"][-1] == latest_user,
            "source_records": len(compiled.records),
            "selected_records": len(candidates),
            "embedding_cache_hits": cache_after["hits"] - cache_before["hits"],
            "embedding_cache_misses": cache_after["misses"] - cache_before["misses"],
        }
        return rewritten

    def after_response(self, body: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        return response
