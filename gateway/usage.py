"""Normalize provider-reported token usage without changing response bytes."""

from __future__ import annotations

import json
from typing import Any


def extract_usage(payload: Any) -> dict[str, int] | None:
    """Return normalized token counts from OpenAI, Anthropic, or Ollama JSON."""
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
    _take(normalized, "input_tokens", payload.get("prompt_eval_count"))
    _take(normalized, "output_tokens", payload.get("eval_count"))
    if "total_tokens" not in normalized and {
        "input_tokens", "output_tokens"
    } <= normalized.keys():
        normalized["total_tokens"] = normalized["input_tokens"] + normalized["output_tokens"]
    return normalized or None


def _take(target: dict[str, int], key: str, value: Any) -> None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        target[key] = max(target.get(key, 0), value)


class StreamUsageAccumulator:
    """Observe SSE or NDJSON chunks while the original bytes pass through unchanged."""

    def __init__(self) -> None:
        self._buffer = b""
        self._usage: dict[str, int] = {}

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
        if {"input_tokens", "output_tokens"} <= self._usage.keys():
            self._usage["total_tokens"] = (
                self._usage["input_tokens"] + self._usage["output_tokens"]
            )
        return dict(self._usage) or None

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
