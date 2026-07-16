"""Normalize provider-reported token usage without changing response bytes."""

from __future__ import annotations

import json
from typing import Any


def extract_usage(payload: Any) -> dict[str, int] | None:
    """Return normalized token and provider prompt-cache counts.

    ``input_tokens`` preserves the provider's native input count. Anthropic
    reports cache creation/read tokens alongside (rather than inside) that
    count, so ``effective_input_tokens`` adds those fields. OpenAI's
    ``prompt_tokens_details.cached_tokens`` is already a subset of
    ``prompt_tokens`` and is therefore observed without adding it again.
    """
    if not isinstance(payload, dict):
        return None
    candidates = [payload]
    message = payload.get("message")
    if isinstance(message, dict):
        candidates.append(message)
    normalized: dict[str, int] = {}
    for candidate in candidates:
        usage = candidate.get("usage")
        if isinstance(usage, dict):
            _take(normalized, "input_tokens", usage.get("prompt_tokens"))
            _take(normalized, "output_tokens", usage.get("completion_tokens"))
            _take(normalized, "input_tokens", usage.get("input_tokens"))
            _take(normalized, "output_tokens", usage.get("output_tokens"))
            _take(normalized, "total_tokens", usage.get("total_tokens"))
            _take(normalized, "cache_creation_input_tokens", usage.get("cache_creation_input_tokens"))
            _take(normalized, "cache_read_input_tokens", usage.get("cache_read_input_tokens"))
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                _take(normalized, "cached_input_tokens", details.get("cached_tokens"))
    _take(normalized, "input_tokens", payload.get("prompt_eval_count"))
    _take(normalized, "output_tokens", payload.get("eval_count"))
    if "cache_read_input_tokens" in normalized:
        normalized["cached_input_tokens"] = normalized["cache_read_input_tokens"]
    additive_cache_usage = (
        "cache_creation_input_tokens" in normalized
        or "cache_read_input_tokens" in normalized
    )
    if "input_tokens" in normalized and additive_cache_usage:
        normalized["effective_input_tokens"] = (
            normalized["input_tokens"]
            + normalized.get("cache_creation_input_tokens", 0)
            + normalized.get("cache_read_input_tokens", 0)
        )
    total_input = normalized.get("effective_input_tokens", normalized.get("input_tokens"))
    if "total_tokens" not in normalized and total_input is not None and "output_tokens" in normalized:
        normalized["total_tokens"] = total_input + normalized["output_tokens"]
    return normalized or None


def _take(target: dict[str, int], key: str, value: Any) -> None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        target[key] = max(target.get(key, 0), value)


class StreamUsageAccumulator:
    """Observe SSE or NDJSON chunks while the original bytes pass through unchanged."""

    def __init__(self) -> None:
        self._buffer = b""
        self._usage: dict[str, int] = {}
        self._text: list[str] = []

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        lines = self._buffer.split(b"\n")
        self._buffer = lines.pop()
        for line in lines:
            self._consume_line(line)

    def finish(self) -> dict[str, int] | None:
        if self._buffer:
            self._consume_line(self._buffer)
            self._buffer = b""
        additive_cache_usage = (
            "cache_creation_input_tokens" in self._usage
            or "cache_read_input_tokens" in self._usage
        )
        if "input_tokens" in self._usage and additive_cache_usage:
            self._usage["effective_input_tokens"] = (
                self._usage["input_tokens"]
                + self._usage.get("cache_creation_input_tokens", 0)
                + self._usage.get("cache_read_input_tokens", 0)
            )
        if "cache_read_input_tokens" in self._usage:
            self._usage["cached_input_tokens"] = self._usage["cache_read_input_tokens"]
        total_input = self._usage.get("effective_input_tokens", self._usage.get("input_tokens"))
        if total_input is not None and "output_tokens" in self._usage:
            self._usage["total_tokens"] = total_input + self._usage["output_tokens"]
        return dict(self._usage) or None

    def text(self) -> str:
        return "".join(self._text)

    def _consume_line(self, line: bytes) -> None:
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line or line == b"[DONE]":
            return
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        usage = extract_usage(payload)
        if usage:
            for key, value in usage.items():
                self._usage[key] = max(self._usage.get(key, 0), value)
        text = _stream_text(payload)
        if text:
            self._text.append(text)


def _stream_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    try:
        value = payload["choices"][0]["delta"]["content"]
        if isinstance(value, str):
            return value
    except (KeyError, IndexError, TypeError):
        pass
    delta = payload.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("text"), str):
        return delta["text"]
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    value = payload.get("response")
    return value if isinstance(value, str) else ""
