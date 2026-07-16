"""Claude Code transcript adapter using the Codex hook processing core."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
import urllib.request
import uuid

MAX_TRANSCRIPT_BYTES = 4 * 1024 * 1024
MAX_MESSAGES = 200
NOISE_PREFIXES = (
    "<environment_context", "<codex_internal_context", "<permissions",
    "<collaboration_mode", "<skills_instructions", "<apps_instructions",
    "<plugins_instructions", "<user_instructions", "<recommended_plugins",
    "# AGENTS.md", "# Instructions",
)
Optimizer = Callable[[list[dict[str, str]], str, int], dict[str, Any]]
EVENT_SCHEMA = "aioptimizer.episode-event.v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,79}$")
_CATEGORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,79}$")
_CATEGORY_KEYS = {
    "error_type", "route", "route_reason", "sidecar_error_type", "sidecar_state",
    "workspace_state",
}
_FORBIDDEN_PARTS = (
    "acceptance", "argument", "body", "command", "content", "message_text",
    "output_text", "producer_result", "prompt", "query", "response_text",
    "secret", "spec", "title", "transcript",
)


def episode_id_from_payload(payload):
    source = payload.get("session_id")
    if not isinstance(source, str) or not source:
        source = payload.get("transcript_path")
    if not isinstance(source, str) or not source:
        return "ep-" + uuid.uuid4().hex[:24]
    import hashlib
    return "ep-" + hashlib.sha256(
        b"aioptimizer-episode-v1\0" + source.encode("utf-8", errors="replace")
    ).hexdigest()[:24]


def new_turn_id():
    return "turn-" + uuid.uuid4().hex[:24]


def _validate_id(value, field):
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{field} must be an opaque identifier")
    return value


def make_event(*, source, episode_id, turn_id, **measurements):
    clean = {}
    for key, value in measurements.items():
        if any(part in key.lower() for part in _FORBIDDEN_PARTS):
            raise ValueError(f"measurement key is not content-free: {key}")
        if isinstance(value, bool) or isinstance(value, int) and not isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            clean[key] = value
        elif key in _CATEGORY_KEYS and isinstance(value, str) and _CATEGORY_RE.fullmatch(value):
            clean[key] = value
        elif value is not None:
            raise ValueError(f"measurement is not content-free: {key}")
    return {
        "schema": EVENT_SCHEMA,
        "event_id": "evt-" + uuid.uuid4().hex,
        "episode_id": _validate_id(episode_id, "episode_id"),
        "turn_id": _validate_id(turn_id, "turn_id"),
        "source": source,
        "event_type": "context_route",
        "ts": time.time(),
        **clean,
    }


def _context_measurements(result):
    values = {}
    for key in (
        "route", "route_reason", "history_chars", "output_chars", "integrity_ok",
        "fail_open", "source_records", "embedding_cache_hits", "embedding_cache_misses",
    ):
        if key in result:
            values[key] = result[key]
    load = result.get("load")
    if isinstance(load, dict):
        for key in (
            "token_like_count", "word_token_count", "unique_token_ratio",
            "compression_ratio", "duplicate_turn_ratio", "max_token_run",
            "max_char_run", "repeated_run_ratio", "turn_count", "candidate_count",
            "recent_candidate_count", "obscured_record_count", "repetitive", "load_pressure",
        ):
            if key in load:
                values[f"load_{key}"] = load[key]
    relevance = result.get("relevance")
    if isinstance(relevance, dict):
        for key in (
            "candidates", "rankable_candidates", "peak", "margin", "mean",
            "best_record_age", "best_record_age_records", "best_in_recent_tail",
            "recent_candidates", "recent_peak",
        ):
            if key in relevance:
                values[f"relevance_{key}"] = relevance[key]
    return values


def extract_visible_messages(path: str | Path) -> list[dict[str, str]]:
    """Read Claude Code conversation rows and retain visible message text."""
    transcript = Path(path)
    if not transcript.is_file():
        raise FileNotFoundError(transcript)
    with transcript.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - MAX_TRANSCRIPT_BYTES)
        stream.seek(start)
        if start:
            stream.readline()
        data = stream.read().decode("utf-8", errors="replace")

    try:
        from aioptimizer.messages import message_text
    except ImportError:
        def message_text(message):
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)

    messages = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed transcript row {line_number}") from error
        if not isinstance(row, dict) or row.get("type") not in {"user", "assistant"}:
            continue
        message = row.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = message_text(message).strip()
        if text and not text.startswith(NOISE_PREFIXES):
            messages.append({"role": role, "content": text})
    return messages[-MAX_MESSAGES:]


def request_context(messages, query, output_budget_chars, *, endpoint, timeout_seconds=30.0,
                    episode_id=None, turn_id=None):
    request_body = {
        "messages": messages,
        "query": query,
        "output_budget_chars": output_budget_chars,
    }
    headers = {"Content-Type": "application/json"}
    if episode_id is not None:
        request_body["episode_id"] = _validate_id(episode_id, "episode_id")
        headers["X-AIOptimizer-Episode-ID"] = episode_id
    if turn_id is not None:
        request_body["turn_id"] = _validate_id(turn_id, "turn_id")
        headers["X-AIOptimizer-Turn-ID"] = turn_id
    body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("context optimizer returned a non-object")
    return result


def process_hook(payload, *, optimizer: Optimizer, output_budget_chars=6000,
                 episode_id=None, turn_id=None):
    started = time.perf_counter()
    episode_id = _validate_id(episode_id or episode_id_from_payload(payload), "episode_id")
    turn_id = _validate_id(turn_id or new_turn_id(), "turn_id")
    prompt = payload.get("prompt")
    transcript_path = payload.get("transcript_path")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(transcript_path, str):
        return None, make_event(
            source="claude_hook", episode_id=episode_id, turn_id=turn_id,
            route="invalid_input", injected=False, latency_ms=0.0,
        )
    messages = extract_visible_messages(transcript_path)
    receipt = {
        "messages": len(messages),
        "history_chars": sum(len(message["content"]) for message in messages),
    }
    try:
        result = optimizer(messages, prompt, output_budget_chars)
        for field, expected in (("episode_id", episode_id), ("turn_id", turn_id)):
            if result.get(field) not in (None, expected):
                raise ValueError(f"context optimizer returned mismatched {field}")
        route = str(result.get("route") or "unknown")
        context = result.get("context")
        receipt.update(_context_measurements(result))
        if route != "attention" or not isinstance(context, str) or not context:
            return None, _finish(receipt, started, episode_id, turn_id)
        additional_context = (
            '<aioptimizer_context route="attention">\n'
            "Locally selected prior context. Source labels refer to visible transcript turns; "
            "the current user request remains authoritative.\n\n"
            f"{context}\n</aioptimizer_context>"
        )
        receipt["injected"] = True
        return {"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit", "additionalContext": additional_context,
        }}, _finish(receipt, started, episode_id, turn_id)
    except Exception as error:
        receipt.update({"route": "error", "error_type": type(error).__name__, "injected": False})
        return None, _finish(receipt, started, episode_id, turn_id)


def _finish(receipt, started, episode_id, turn_id):
    receipt.setdefault("injected", False)
    receipt["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return make_event(
        source="claude_hook", episode_id=episode_id, turn_id=turn_id, **receipt
    )


def append_receipt(cwd: str | Path, receipt: dict[str, Any]) -> None:
    target = Path(cwd) / ".aioptimizer" / "codex_hook_ledger.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    row = dict(receipt)
    row.setdefault("ts", time.time())
    payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError("hook receipt append was incomplete")
    finally:
        os.close(descriptor)
