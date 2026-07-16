#!/usr/bin/env python3
"""run_loop — launch the live scheduler loop over the current board.

Fable populates the board (`board.py add ...`); this drives it with real executors:
ShellExecutor runs `test`/`run` acceptance commands locally, CodexExecutor (opt-in) runs
`impl` tasks via `codex exec`, and verdict/fable tasks are parked for the human (NeedsHuman).

Safe defaults: Codex and legacy shell execution are OFF, typed commands require an explicit
argv-prefix policy, and a `--budget` cost ceiling caps spend.
Independent-review failures stay terminal by default. Set
`--verification-retries N` to allow at most N evidence-preserving repair attempts;
the rejection note is supplied to the next Codex execution.

    # local-only loop (no Codex): runs command tasks, parks impl/verdict
    python agent_bus/run_loop.py --budget 5 --allow-command-prefix '["python","-m","unittest"]'

    # with Codex as a real core (in an environment where `codex` is on PATH + auto-approve):
    python agent_bus/run_loop.py --budget 5 --codex --codex-args "--full-auto"
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_bus.board import Board
from agent_bus.executors import CodexExecutor, CommandPolicy, ExecutorVerifier, RoutingExecutor, ShellExecutor
from agent_bus.scheduler import GitCommitter, Scheduler, _summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent), help="Bus/board directory.")
    p.add_argument("--repo", default=str(_REPO_ROOT), help="Working dir for executors (repo root).")
    p.add_argument("--budget", type=float, default=5.0, help="Cost ceiling (maskable interrupt).")
    p.add_argument("--max-ticks", type=int, default=50)
    p.add_argument("--verification-retries", type=int, default=0,
                   help="Bounded repair attempts after independent review rejects a task.")
    p.add_argument("--codex", action="store_true", help="Enable Codex as a real core (needs codex on PATH).")
    p.add_argument("--codex-bin", default="codex")
    p.add_argument("--codex-args", default="", help="Extra args passed to `codex exec` (e.g. --full-auto).")
    p.add_argument("--commit", action="store_true", help="git commit locally on each retired task (durable, recoverable; never pushes).")
    p.add_argument("--allow-command-prefix", action="append", default=[], help='Allowed typed argv prefix as JSON, e.g. ["python","-m","unittest"]. Repeatable.')
    p.add_argument("--allow-legacy-shell", action="store_true", help="Compatibility escape hatch for old acceptance strings (unsafe for model-authored tasks).")
    args = p.parse_args()

    prefixes = [json.loads(value) for value in args.allow_command_prefix]
    policy = CommandPolicy(prefixes)
    shell = ShellExecutor(cwd=args.repo, command_policy=policy, allow_legacy_shell=args.allow_legacy_shell)
    codex = CodexExecutor(args.repo, codex=args.codex_bin, extra_args=shlex.split(args.codex_args), command_policy=policy, allow_legacy_shell=args.allow_legacy_shell) if args.codex else None
    router = RoutingExecutor(shell=shell, codex=codex)
    verifier = ExecutorVerifier(shell, reviewer="shell-verifier")
    on_retire = GitCommitter(args.repo) if args.commit else None
    sched = Scheduler(
        Path(args.root),
        executor=router,
        verifier=verifier,
        budget=args.budget,
        on_retire=on_retire,
        workspace_root=Path(args.repo),
        max_verification_retries=args.verification_retries,
    )

    history = sched.run(max_ticks=args.max_ticks)
    summary = _summarize(history)
    print("loop summary:", summary)

    board = Board(Path(args.root))
    awaiting = [t for t in board.all() if t["state"] == "blocked"]
    if awaiting:
        print("\nawaiting the human (Fable) -- resolve these to unblock retirement:")
        for t in awaiting:
            print(f"  {t['id']} [{t['op']}/{t['tier']}] {t['title']}")
    if summary["tripped"]:
        print("\n[!] cost ceiling tripped -- raise --budget or trim the board to continue.")


if __name__ == "__main__":
    main()
