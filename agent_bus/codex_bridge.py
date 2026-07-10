#!/usr/bin/env python3
"""codex_bridge — the Codex-side wire.

Launch this ONCE in an environment where the `codex` binary is on PATH with
auto-approve configured. It polls the board; whenever Fable has queued a ready
Codex-tier task, it invokes `codex exec` to handle it, then loops. That closes the
loop: Fable posts (`board.py add --tier codex ...`) -> this bridge fires Codex ->
Codex reads the bus/board, does the work, commits its declared writes, and posts a
result -> Fable sees it (via the bus and `read_codex.py`).

Fable cannot run this itself (the `codex` binary isn't on Fable's shells) — that is
why it is a small standalone launcher for the human/Codex side. Ctrl-C to stop.

    python agent_bus/codex_bridge.py                 # poll every 180s, `codex` on PATH
    python agent_bus/codex_bridge.py --interval 120 --codex-arg --full-auto
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_bus.board import Board  # noqa: E402

PROMPT = (
    "Per AGENTS.md: read the agent bus (python agent_bus/bus.py read --for codex --new) and the "
    "board (python agent_bus/board.py view). Take the next ready Codex-tier task, implement it, run "
    "its acceptance command, commit only your declared writes, and post a result to fable on the bus. "
    "Do NOT write verdicts or edit memory/PREREGISTRATION/PROJECT_PLAN. If nothing is actionable, "
    "post a one-line status and stop."
)


def ready_codex_tasks(root: str) -> list[str]:
    return [t["id"] for t in Board(root).all() if t["state"] == "ready" and t["tier"] == "codex"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent), help="Board directory.")
    ap.add_argument("--interval", type=float, default=180.0, help="Seconds between board polls.")
    ap.add_argument("--codex", default="codex", help="Path to the codex binary.")
    ap.add_argument("--codex-arg", action="append", default=[], help="Extra arg to `codex exec` (repeatable), e.g. --full-auto.")
    ap.add_argument("--once", action="store_true", help="Check once and exit (for testing).")
    args = ap.parse_args()

    print(f"codex_bridge: polling {args.root} every {args.interval:.0f}s (Ctrl-C to stop)")
    while True:
        try:
            ids = ready_codex_tasks(args.root)
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] board read error: {exc}")
            ids = []
        if ids:
            print(f"[bridge] {len(ids)} ready Codex task(s) {ids} -> codex exec")
            cmd = [args.codex, "exec", *args.codex_arg, PROMPT]
            try:
                subprocess.run(cmd)
            except FileNotFoundError:
                print(f"[bridge] codex binary '{args.codex}' not found — run this where codex is on PATH.")
                return
        else:
            print("[bridge] no ready Codex task")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
