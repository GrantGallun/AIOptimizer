"""Claude Code transcript adapter using the Codex hook processing core."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable
import urllib.request

MAX_TRANSCRIPT_BYTES = 4 * 1024 * 1024
MAX_MESSAGES = 200
NOISE_PREFIXES = (
    "<environment_context", "<codex_internal_context", "<permissions",
    "<collaboration_mode", "<skills_instructions", "<apps_instructions",
    "<plugins_instructions", "<user_instructions", "<recommended_plugins",
    "# AGENTS.md", "# Instructions",
)
Optimizer = Callable[[list[dict[str, str]], str, int], dict[str, Any]]


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


def request_context(messages, query, output_budget_chars, *, endpoint, timeout_seconds=30.0):
    body = json.dumps({
        "messages": messages,
        "query": query,
        "output_budget_chars": output_budget_chars,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise ValueError("context optimizer returned a non-object")
    return result


def process_hook(payload, *, optimizer: Optimizer, output_budget_chars=6000):
    started = time.perf_counter()
    prompt = payload.get("prompt")
    transcript_path = payload.get("transcript_path")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(transcript_path, str):
        return None, {"route": "invalid_input", "latency_ms": 0.0}
    messages = extract_visible_messages(transcript_path)
    receipt = {
        "query_sha256_16": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
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
            return None, _finish(receipt, started)
        additional_context = (
            '<aioptimizer_context route="attention">\n'
            "Locally selected prior context. Source labels refer to visible transcript turns; "
            "the current user request remains authoritative.\n\n"
            f"{context}\n</aioptimizer_context>"
        )
        receipt["injected"] = True
        return {"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit", "additionalContext": additional_context,
        }}, _finish(receipt, started)
    except Exception as error:
        receipt.update({"route": "error", "error_type": type(error).__name__, "injected": False})
        return None, _finish(receipt, started)


def _finish(receipt, started):
    receipt.setdefault("injected", False)
    receipt["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return receipt


def append_receipt(cwd: str | Path, receipt: dict[str, Any]) -> None:
    target = Path(cwd) / ".aioptimizer" / "codex_hook_ledger.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"ts": time.time(), **receipt}, ensure_ascii=False, separators=(",", ":")) + "\n")
