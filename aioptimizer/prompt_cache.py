"""Content-free diagnostics for provider prompt-prefix reuse opportunities.

This middleware never mutates a request and never claims a provider cache hit.
It fingerprints the stable leading portion of the final upstream prompt so the
ledger can distinguish actual provider cache receipts from prompts that merely
had a repeatable prefix candidate.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any


def stable_prefix(body: dict[str, Any]) -> dict[str, Any] | None:
    """Return the provider-visible static prefix, preserving key order.

    Tools and top-level system instructions precede messages for Anthropic.
    For OpenAI-compatible messages, only the initial contiguous system/developer
    block is considered static. Dynamic user history is deliberately excluded.
    """
    prefix: dict[str, Any] = {}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        prefix["tools"] = tools
    if "system" in body and body["system"] not in (None, "", []):
        prefix["system"] = body["system"]

    messages = body.get("messages")
    message_prefix = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system", "developer"
            }:
                break
            message_prefix.append(message)
    if message_prefix:
        prefix["messages"] = message_prefix

    # Structured-output schemas are inserted into the prompt prefix by some
    # providers and must remain identical to be reusable.
    if "response_format" in body and body["response_format"] is not None:
        prefix["response_format"] = body["response_format"]
    return prefix or None


class PromptCacheTelemetryMiddleware:
    """Record repeatable static-prefix candidates without changing traffic."""

    def __init__(self, *, max_prefixes: int = 1024) -> None:
        if (
            not isinstance(max_prefixes, int)
            or isinstance(max_prefixes, bool)
            or max_prefixes < 1
        ):
            raise ValueError("max_prefixes must be a positive integer")
        self.max_prefixes = max_prefixes
        self._counts: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()
        self._local = threading.local()
        self._observations = 0
        self._candidate_reuses = 0

    def before_request(self, body: dict[str, Any]) -> dict[str, Any]:
        prefix = stable_prefix(body)
        if prefix is None:
            self._local.receipt = {"observed": False, "candidate_reuse": False}
            return body
        serialized = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
        fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        with self._lock:
            previous = self._counts.get(fingerprint, 0)
            self._counts[fingerprint] = previous + 1
            self._counts.move_to_end(fingerprint)
            while len(self._counts) > self.max_prefixes:
                self._counts.popitem(last=False)
            self._observations += 1
            self._candidate_reuses += int(previous > 0)
        self._local.receipt = {
            "observed": True,
            "candidate_reuse": previous > 0,
            "prefix_chars": len(serialized),
            "prefix_fingerprint": fingerprint,
        }
        return body

    def after_response(self, body: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        return response

    def receipt_metadata(self) -> dict[str, Any]:
        return dict(getattr(
            self._local,
            "receipt",
            {"observed": False, "candidate_reuse": False},
        ))

    def status_metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "observations": self._observations,
                "candidate_reuses": self._candidate_reuses,
                "unique_prefixes": len(self._counts),
                "max_prefixes": self.max_prefixes,
                "mutates_requests": False,
            }
