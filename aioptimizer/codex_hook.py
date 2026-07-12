"""Subscription-safe Codex hook adapter for the local deterministic compiler."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable
import urllib.request

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
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    payload = json.dumps({
        "messages": messages,
        "query": query,
        "output_budget_chars": output_budget_chars,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
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
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return Codex hook output plus a private, content-free receipt."""
    started = time.perf_counter()
    prompt = payload.get("prompt")
    transcript_path = payload.get("transcript_path")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(transcript_path, str):
        return None, {"route": "invalid_input", "latency_ms": 0.0}
    messages = extract_visible_messages(transcript_path)
    query_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    receipt: dict[str, Any] = {
        "query_sha256_16": query_hash,
        "messages": len(messages),
        "history_chars": sum(len(message["content"]) for message in messages),
    }
    try:
        result = optimizer(messages, prompt, output_budget_chars)
        route = str(result.get("route") or "unknown")
        context = result.get("context")
        receipt.update({
            "route": route,
            "output_chars": int(result.get("output_chars", 0)),
            "embedding_cache_hits": int(result.get("embedding_cache_hits", 0)),
            "embedding_cache_misses": int(result.get("embedding_cache_misses", 0)),
        })
        if route != "attention" or not isinstance(context, str) or not context:
            return None, _finish_receipt(receipt, started)
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
        return output, _finish_receipt(receipt, started)
    except Exception as error:  # hook must fail open; details stay local and bounded
        receipt.update({
            "route": "error",
            "error_type": type(error).__name__,
            "injected": False,
        })
        return None, _finish_receipt(receipt, started)


def _finish_receipt(receipt: dict[str, Any], started: float) -> dict[str, Any]:
    receipt.setdefault("injected", False)
    receipt["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return receipt


def append_receipt(cwd: str | Path, receipt: dict[str, Any]) -> None:
    target = Path(cwd) / ".aioptimizer" / "codex_hook_ledger.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), **receipt}
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
