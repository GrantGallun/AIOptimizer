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

    python agent_bus/codex_bridge.py                 # poll every 180s
    python agent_bus/codex_bridge.py --interval 30   # faster polling
    python agent_bus/codex_bridge.py --diagnose

Failure handling (added 2026-07-17 after a real hot loop): a corrupt `~/.codex/rules`
file made every `codex exec` exit 1 at startup, and the bridge re-dispatched the same
ready task every poll forever — no failure count, no backoff, no ceiling. The board has
a watchdog for the dual problem (a crashed core deadlocking a *claimed* task) but had
nothing for the inverse (a broken core hot-looping a *ready* one). `DispatchGuard` is
that missing half, and the scheduler's `Governor` now bounds spend here too — the bridge
is the only component that spawns paid processes in an unbounded loop, and it was the
only one with no budget ceiling.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_bus.board import Board  # noqa: E402
from agent_bus.cache import Cache  # noqa: E402
from agent_bus.scheduler import GitCommitter, Governor  # noqa: E402

PROMPT_TEMPLATE = (
    "Per AGENTS.md: read the agent bus (python agent_bus/bus.py read --for codex --new) and the "
    "board (python agent_bus/board.py view). Work on Codex task {task_id} only: claim it, implement it, run "
    "its acceptance command, submit it, and post a result to fable on the bus. Do not run git add/commit; "
    "the trusted bridge commits only the task's declared writes after this process exits. "
    "Do NOT write verdicts or edit memory/PREREGISTRATION/PROJECT_PLAN. If nothing is actionable, "
    "post a one-line status and stop."
)


class BridgeGovernor(Governor):
    """The scheduler's budget ceiling, published under bridge-owned keys.

    The base class publishes as ``writer="fable"`` on ``governor.*``; the bridge is a
    separate driver, so it would both misattribute the writer and contend for the same
    cache lines if a scheduler ran alongside it. Same accounting, different lines.
    """

    def _publish(self) -> None:
        self.cache.set("governor.bridge.budget", f"{self.budget:.2f}", writer="codex", scope="governor")
        self.cache.set("governor.bridge.spent", f"{self.spent:.2f}", writer="codex", scope="governor")


class DispatchGuard:
    """Circuit breaker: stop re-dispatching work that keeps failing.

    Two independent counters, because the bridge cannot otherwise tell *this task is
    bad* from *codex itself is broken*:

    * per-task consecutive failures -> exponential backoff, then quarantine. A task
      whose acceptance never passes stops costing anything after ``max_task_failures``.
    * global consecutive failures (any task, reset by any success) -> ``harness_suspect``.
      When nothing at all is succeeding, the fault is almost certainly the harness
      (corrupt rules file, revoked auth, bad sandbox flag) and retrying any task is
      waste, so the caller halts instead of spinning.

    ``max_global_failures <= max_task_failures`` matters and is not arbitrary. With a
    single ready task the two counters advance together, so a *lower or equal* global
    limit makes the harness verdict win the tie. Otherwise the task would be quarantined
    first and a broken harness would get silently misattributed to the one good task in
    the queue — which is exactly what the 2026-07-17 failure looked like from outside.

    Deliberately model-free and probe-free: a cheap "is codex healthy" preflight would
    have to guess which invocation loads the config that broke, so we infer it from the
    failure pattern we can actually observe.
    """

    def __init__(
        self,
        *,
        max_task_failures: int = 3,
        max_global_failures: int = 3,
        backoff_base: float = 30.0,
        backoff_cap: float = 900.0,
        clock=time.monotonic,
    ) -> None:
        if max_task_failures < 1 or max_global_failures < 1:
            raise ValueError("failure limits must be >= 1")
        if backoff_base <= 0 or backoff_cap < backoff_base:
            raise ValueError("backoff_base must be > 0 and backoff_cap >= backoff_base")
        self.max_task_failures = max_task_failures
        self.max_global_failures = max_global_failures
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.clock = clock
        self.consecutive_failures = 0
        self._failures: dict[str, int] = {}
        self._next_attempt: dict[str, float] = {}
        self._quarantined: set[str] = set()

    def is_quarantined(self, task_id: str) -> bool:
        return task_id in self._quarantined

    def blocked_reason(self, task_id: str) -> str | None:
        """Why this task must not be dispatched right now, or None to go ahead."""
        if task_id in self._quarantined:
            return f"quarantined after {self._failures[task_id]} consecutive failures"
        wait = self._next_attempt.get(task_id, 0.0) - self.clock()
        if wait > 0:
            return f"backing off {wait:.0f}s ({self._failures.get(task_id, 0)} failures)"
        return None

    def record_success(self, task_id: str) -> None:
        self._failures.pop(task_id, None)
        self._next_attempt.pop(task_id, None)
        self.consecutive_failures = 0

    def record_failure(self, task_id: str) -> int:
        count = self._failures.get(task_id, 0) + 1
        self._failures[task_id] = count
        self.consecutive_failures += 1
        if count >= self.max_task_failures:
            self._quarantined.add(task_id)
            self._next_attempt.pop(task_id, None)
        else:
            delay = min(self.backoff_base * (2 ** (count - 1)), self.backoff_cap)
            self._next_attempt[task_id] = self.clock() + delay
        return count

    def harness_suspect(self) -> bool:
        return self.consecutive_failures >= self.max_global_failures


def ready_codex_tasks(root: str) -> list[str]:
    return [t["id"] for t in Board(root).all() if t["state"] == "ready" and t["tier"] == "codex"]


def publish_presence(cache: Cache, value: str) -> None:
    """Write bridge liveness to shared state without making polling depend on it."""
    try:
        cache.set("status.codex", value, writer="codex", scope="shared")
    except Exception as exc:  # noqa: BLE001
        print(f"[bridge] presence update failed: {exc}")


def build_codex_command(
    codex: str,
    *,
    root: str | Path,
    extra_args: list[str],
    task_id: str,
) -> list[str]:
    """Build one sandboxed headless process for exactly one board task."""
    repo = Path(root).resolve().parent
    command = [codex, *extra_args, "-a", "never", "-s", "workspace-write", "-C", str(repo)]
    command.extend(["exec", PROMPT_TEMPLATE.format(task_id=task_id)])
    return command


def commit_completed_task(root: str | Path, task_id: str) -> str | None:
    """Commit one completed task's declared paths outside the model sandbox."""
    board = Board(Path(root))
    task = board.get(task_id)
    if task is None or task["state"] not in {"executed", "verified", "retired"}:
        return None
    if not task.get("writes"):
        print(f"[bridge] {task_id} completed without declared writes; not committing")
        return None
    revision = GitCommitter(Path(root).resolve().parent)(task)
    if revision:
        print(f"[bridge] committed {task_id} declared writes as {revision}")
    else:
        print(f"[bridge] {task_id} has no uncommitted declared-path changes")
    return revision


def resolve_codex(explicit: str | None = None, *, environ: dict[str, str] | None = None) -> str | None:
    """Resolve Codex without requiring a session-scoped PowerShell variable."""
    env = os.environ if environ is None else environ
    requested = explicit or env.get("CODEX_BIN")
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        found = shutil.which(requested)
        return str(Path(found).resolve()) if found else None

    found = shutil.which("codex")
    if found:
        return str(Path(found).resolve())

    local_app_data = env.get("LOCALAPPDATA")
    if local_app_data:
        install_root = Path(local_app_data) / "OpenAI" / "Codex" / "bin"
        candidates = [path for path in install_root.glob("*/codex.exe") if path.is_file()]
        if candidates:
            newest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
            return str(newest.resolve())
    return None


def main() -> None:
    # Same hole bus.py hit (fixed in 4eae2ad): bridge output carries em-dashes/arrows, and a
    # cp1252 Windows console raises UnicodeEncodeError on them — which would crash the bridge
    # inside its own failure reporting, exactly when you need to read it.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent), help="Board directory.")
    ap.add_argument("--interval", type=float, default=180.0, help="Seconds between board polls.")
    ap.add_argument("--heartbeat", type=float, default=300.0, help="Seconds between shared-cache presence updates.")
    ap.add_argument("--codex", default=None, help="Optional Codex path/name override.")
    ap.add_argument("--codex-arg", action="append", default=[], help="Extra arg to `codex exec` (repeatable), e.g. --full-auto.")
    ap.add_argument("--once", action="store_true", help="Check once and exit (for testing).")
    ap.add_argument("--diagnose", action="store_true", help="Resolve Codex and exit without polling or claiming work.")
    ap.add_argument("--budget-seconds", type=float, default=3600.0,
                    help="Ceiling on cumulative Codex process wall-clock. Halts the bridge when reached.")
    ap.add_argument("--max-task-failures", type=int, default=3,
                    help="Consecutive failures before a task is quarantined (stops being dispatched).")
    ap.add_argument("--max-global-failures", type=int, default=3,
                    help="Consecutive failures across all tasks before the bridge halts (harness likely broken). "
                         "Keep <= --max-task-failures so a broken harness outranks blaming one task.")
    args = ap.parse_args()

    codex = resolve_codex(args.codex)
    if codex is None:
        requested = args.codex or os.environ.get("CODEX_BIN") or "automatic discovery"
        print(f"codex_bridge: could not resolve Codex via {requested}")
        raise SystemExit(2)
    args.codex = codex
    print(f"codex_bridge: using {codex}")
    if args.diagnose:
        return
    if args.interval <= 0 or args.heartbeat <= 0:
        ap.error("--interval and --heartbeat must be positive")
    if args.budget_seconds <= 0:
        ap.error("--budget-seconds must be positive")
    cache = Cache(Path(args.root))
    guard = DispatchGuard(
        max_task_failures=args.max_task_failures,
        max_global_failures=args.max_global_failures,
    )
    governor = BridgeGovernor(cache, args.budget_seconds)
    publish_presence(cache, "bridge online; starting poll loop")
    last_heartbeat = time.monotonic()
    print(f"codex_bridge: polling {args.root} every {args.interval:.0f}s (Ctrl-C to stop)")
    print(f"codex_bridge: budget {args.budget_seconds:.0f}s of Codex process time; "
          f"quarantine after {args.max_task_failures} task failures; "
          f"halt after {args.max_global_failures} consecutive failures")
    while True:
        try:
            ids = ready_codex_tasks(args.root)
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] board read error: {exc}")
            ids = []
        dispatchable = []
        for task_id in ids:
            reason = guard.blocked_reason(task_id)
            if reason is None:
                dispatchable.append(task_id)
            else:
                print(f"[bridge] skipping {task_id}: {reason}")
        if dispatchable:
            publish_presence(cache, f"bridge online; dispatching Codex tasks: {','.join(dispatchable)}")
            print(f"[bridge] {len(dispatchable)} ready Codex task(s) {dispatchable}; dispatching one process per task")
            for task_id in dispatchable:
                if governor.tripped():
                    break
                cmd = build_codex_command(
                    args.codex,
                    root=args.root,
                    extra_args=args.codex_arg,
                    task_id=task_id,
                )
                started = time.monotonic()
                try:
                    completed = subprocess.run(cmd)
                except FileNotFoundError:
                    print(f"[bridge] resolved Codex binary disappeared: {args.codex}")
                    return
                finally:
                    governor.charge(max(time.monotonic() - started, 0.0))
                if completed.returncode == 0:
                    guard.record_success(task_id)
                    commit_completed_task(args.root, task_id)
                else:
                    count = guard.record_failure(task_id)
                    print(f"[bridge] {task_id} Codex process exited {completed.returncode}; "
                          f"not committing (failure {count}/{args.max_task_failures})")
                    # Harness suspicion outranks quarantine: if nothing is succeeding, the task is
                    # probably innocent and quarantining it would blame the wrong thing.
                    if guard.harness_suspect():
                        print(f"[bridge] HALTING: {guard.consecutive_failures} consecutive failures with no "
                              f"success — this looks like a broken harness (corrupt ~/.codex/rules, revoked "
                              f"auth, bad sandbox flag), not a bad task. Retrying would just spin. "
                              f"Check the Codex output above, fix it, and restart the bridge.")
                        publish_presence(cache, "bridge halted; harness suspected broken")
                        raise SystemExit(1)
                    if guard.is_quarantined(task_id):
                        print(f"[bridge] {task_id} QUARANTINED after {count} consecutive failures; "
                              f"it will not be dispatched again this run. Fix it or squash it.")
            if governor.tripped():
                print(f"[bridge] HALTING: budget ceiling reached "
                      f"({governor.spent:.0f}s/{governor.budget:.0f}s of Codex process time). "
                      f"Raise --budget-seconds to continue.")
                publish_presence(cache, "bridge halted; budget ceiling reached")
                raise SystemExit(3)
            publish_presence(cache, "bridge online; Codex dispatch returned; polling")
        elif ids:
            print(f"[bridge] {len(ids)} ready task(s) but all are backing off or quarantined")
        else:
            print("[bridge] no ready Codex task")
            now = time.monotonic()
            if now - last_heartbeat >= args.heartbeat:
                publish_presence(cache, "bridge online/idle; no ready Codex tasks")
                last_heartbeat = now
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
