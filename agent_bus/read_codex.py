#!/usr/bin/env python3
"""read_codex — Fable's read-wire into Codex.

Codex (OpenAI CLI) stores each session as a rollout JSONL under
``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl``. This surfaces recent Codex
activity (user turns, Codex replies, tool calls) so Fable can *see* what Codex is
doing directly, instead of relying on the human to relay or on Codex posting to
the bus. Read-only; no dependency on Codex being invocable from here.

    python agent_bus/read_codex.py                 # last 20 msgs of the newest session
    python agent_bus/read_codex.py --sessions 2 --tail 40
    python agent_bus/read_codex.py --grep langgraph --full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SESSIONS = Path.home() / ".codex" / "sessions"
NOISE_PREFIXES = ("<recommended_plugins", "<environment", "<user_instructions", "<plugins", "<system", "# Instructions")


def latest_rollouts(n: int) -> list[Path]:
    if not SESSIONS.exists():
        return []
    files = sorted(SESSIONS.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def extract(path: Path) -> list[tuple[str, str]]:
    """Return (who, text) for message + tool-call entries, in order."""
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("type") != "response_item":
            continue
        p = o.get("payload", {})
        ptype = p.get("type")
        if ptype == "message":
            role = p.get("role")
            if role not in ("user", "assistant"):
                continue
            text = " ".join(c.get("text", "") for c in p.get("content", []) if isinstance(c, dict)).strip()
            if not text or text.startswith(NOISE_PREFIXES):
                continue
            out.append(("USER" if role == "user" else "CODEX", text))
        elif ptype == "function_call":
            name = p.get("name", "?")
            out.append(("  tool", f"{name}(…)"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=1, help="How many recent session files to read.")
    ap.add_argument("--tail", type=int, default=20, help="Show the last N entries.")
    ap.add_argument("--grep", default=None, help="Only entries containing this (case-insensitive).")
    ap.add_argument("--full", action="store_true", help="Do not truncate message text.")
    args = ap.parse_args()

    files = latest_rollouts(args.sessions)
    if not files:
        print(f"no Codex sessions found under {SESSIONS}")
        return
    entries: list[tuple[str, str]] = []
    for f in reversed(files):  # oldest of the selected first
        entries.extend(extract(f))
    if args.grep:
        needle = args.grep.lower()
        entries = [(w, t) for w, t in entries if needle in t.lower()]
    for who, text in entries[-args.tail:]:
        body = text if args.full else (text[:500] + ("…" if len(text) > 500 else ""))
        print(f"--- {who} ---\n{body}\n")


if __name__ == "__main__":
    main()
