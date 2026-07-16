"""Subscription-safe Codex hook adapter for the local deterministic compiler."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import urllib.request

from .episodes import (
    context_result_measurements,
    episode_id_from_payload,
    make_event,
    new_turn_id,
    validate_event_id,
)

DEFAULT_CONTEXT_URL = "http://127.0.0.1:8800/optimize/context"
MAX_TRANSCRIPT_BYTES = 4 * 1024 * 1024
MAX_MESSAGES = 200
NOISE_PREFIXES = (
    "<environment_context",
    "<codex_internal_context",
    "<permissions",
    "<collaboration_mode",
    "<skills_instructions",
    "<apps_instructions",
    "<plugins_instructions",
    "<user_instructions",
    "<recommended_plugins",
    "# AGENTS.md",
    "# Instructions",
)

Optimizer = Callable[[list[dict[str, str]], str, int], dict[str, Any]]


def extract_visible_messages(path: str | Path) -> list[dict[str, str]]:
    """Read a bounded rollout tail and retain visible user/assistant messages only."""
    transcript = Path(path)
    if not transcript.is_file():
        return []
    with transcript.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size - MAX_TRANSCRIPT_BYTES)
        stream.seek(start)
        if start:
            stream.readline()  # discard one partial JSONL row
        data = stream.read().decode("utf-8", errors="replace")
    messages = []
    for line in data.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "response_item":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = "\n".join(
            part.get("text", "")
            for part in payload.get("content", [])
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ).strip()
        if not text or text.startswith(NOISE_PREFIXES):
            continue
        messages.append({"role": role, "content": text})
    return messages[-MAX_MESSAGES:]


def request_context(
    messages: list[dict[str, str]],
    query: str,
    output_budget_chars: int,
    *,
    endpoint: str = DEFAULT_CONTEXT_URL,
    timeout_seconds: float = 30.0,
    episode_id: str | None = None,
    turn_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": messages,
        "query": query,
        "output_budget_chars": output_budget_chars,
    }
    headers = {"Content-Type": "application/json"}
    if episode_id is not None:
        body["episode_id"] = validate_event_id(episode_id, "episode_id")
        headers["X-AIOptimizer-Episode-ID"] = episode_id
    if turn_id is not None:
        body["turn_id"] = validate_event_id(turn_id, "turn_id")
        headers["X-AIOptimizer-Turn-ID"] = turn_id
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("context optimizer returned a non-object")
    return result


def process_hook(
    payload: dict[str, Any],
    *,
    optimizer: Optimizer,
    output_budget_chars: int = 6_000,
    episode_id: str | None = None,
    turn_id: str | None = None,
    source: str = "codex_hook",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return Codex hook output plus a private, content-free receipt."""
    started = time.perf_counter()
    episode_id = validate_event_id(
        episode_id or episode_id_from_payload(payload), "episode_id"
    )
    turn_id = validate_event_id(turn_id or new_turn_id(), "turn_id")
    prompt = payload.get("prompt")
    transcript_path = payload.get("transcript_path")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(transcript_path, str):
        return None, make_event(
            source=source, event_type="context_route", episode_id=episode_id,
            turn_id=turn_id, route="invalid_input", injected=False, latency_ms=0.0,
        )
    messages = extract_visible_messages(transcript_path)
    receipt: dict[str, Any] = {
        "messages": len(messages),
        "history_chars": sum(len(message["content"]) for message in messages),
    }
    try:
        result = optimizer(messages, prompt, output_budget_chars)
        for field, expected in (("episode_id", episode_id), ("turn_id", turn_id)):
            returned = result.get(field)
            if returned is not None and returned != expected:
                raise ValueError(f"context optimizer returned mismatched {field}")
        route = str(result.get("route") or "unknown")
        context = result.get("context")
        receipt.update(context_result_measurements(result))
        if route != "attention" or not isinstance(context, str) or not context:
            return None, _finish_receipt(
                receipt, started, source=source, episode_id=episode_id, turn_id=turn_id
            )
        additional_context = (
            "<aioptimizer_context route=\"attention\">\n"
            "Locally selected prior context. Source labels refer to visible transcript turns; "
            "the current user request remains authoritative.\n\n"
            f"{context}\n"
            "</aioptimizer_context>"
        )
        output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": additional_context,
            }
        }
        receipt["injected"] = True
        return output, _finish_receipt(
            receipt, started, source=source, episode_id=episode_id, turn_id=turn_id
        )
    except Exception as error:  # hook must fail open; details stay local and bounded
        receipt.update({
            "route": "error",
            "error_type": type(error).__name__,
            "injected": False,
        })
        return None, _finish_receipt(
            receipt, started, source=source, episode_id=episode_id, turn_id=turn_id
        )


def _finish_receipt(
    receipt: dict[str, Any],
    started: float,
    *,
    source: str,
    episode_id: str,
    turn_id: str,
) -> dict[str, Any]:
    receipt.setdefault("injected", False)
    receipt["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return make_event(
        source=source, event_type="context_route", episode_id=episode_id,
        turn_id=turn_id, **receipt,
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
