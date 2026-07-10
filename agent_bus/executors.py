#!/usr/bin/env python3
"""executors — real cores behind the scheduler's Executor interface.

`SimExecutor` (in scheduler.py) fakes work for dry runs. These are the real ones:

  * `ShellExecutor` — runs a task's `acceptance` shell command; ok = exit 0. Model-free and
    fully local, so it drives `test`/`run` tasks today (including Qwen-backed evals invoked as
    commands). This is the executor that can run right now in any environment.
  * `CodexExecutor` — shells out to `codex exec` for `impl` tasks. Requires the `codex` binary on
    PATH with auto-approve configured (`~/.codex/config.toml`). Real, but only where Codex is
    reachable — the machine's own shells, not necessarily a sandboxed sub-process elsewhere.
  * `RoutingExecutor` — routes each task to the right core by tier/op, and raises `NeedsHuman`
    for `verdict`/`fable` tasks so the loop hands judgment back (the human/Fable interrupt).

Every `execute` returns `(ok: bool, result: str, cost: float)` — the same contract as SimExecutor.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class NeedsHuman(Exception):
    """The task requires Fable/human judgment (a verdict) and cannot be auto-executed."""


def _tail(text: str, n: int = 3) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " / ".join(lines[-n:]) if lines else ""


class ShellExecutor:
    """Run a task's acceptance command. ok = exit 0. No model, fully local."""

    def __init__(
        self,
        cwd: str | Path,
        *,
        cost: float = 0.02,
        timeout: int = 600,
        allowlist: list[str] | None = None,
    ) -> None:
        self.cwd = str(cwd)
        self.cost = cost
        self.timeout = timeout
        self.allowlist = allowlist

    def execute(self, task: dict[str, Any]) -> tuple[bool, str, float]:
        cmd = task.get("acceptance") or ""
        if not cmd:
            return False, "no acceptance command to run", 0.0
        if self.allowlist is not None:
            stripped = cmd.strip()
            if not any(stripped.startswith(prefix) for prefix in self.allowlist):
                return False, "blocked: command not allowlisted", 0.0
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {self.timeout}s", self.cost
        ok = proc.returncode == 0
        return ok, f"exit {proc.returncode}: {_tail(proc.stdout + proc.stderr)}", self.cost


class CodexExecutor:
    """Drive `codex exec` for an impl task. Requires codex on PATH + auto-approve configured."""

    def __init__(self, cwd: str | Path, *, codex: str = "codex", extra_args: list[str] | None = None, cost: float = 0.4, timeout: int = 1800) -> None:
        self.cwd = str(cwd)
        self.codex = codex
        self.extra_args = extra_args or []  # e.g. ["--full-auto"] or sandbox flags
        self.cost = cost
        self.timeout = timeout

    def execute(self, task: dict[str, Any]) -> tuple[bool, str, float]:
        prompt = task.get("spec") or task.get("title") or ""
        cmd = [self.codex, "exec", *self.extra_args, prompt]
        try:
            proc = subprocess.run(cmd, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            return False, f"codex binary '{self.codex}' not found on PATH", 0.0
        except subprocess.TimeoutExpired:
            return False, f"codex exec timeout after {self.timeout}s", self.cost
        ok = proc.returncode == 0
        # If the task carries an acceptance check, the code must also pass it.
        acceptance = task.get("acceptance")
        if ok and acceptance:
            check = subprocess.run(acceptance, shell=True, cwd=self.cwd, capture_output=True, text=True)
            ok = check.returncode == 0
            return ok, f"codex exit 0; acceptance exit {check.returncode}: {_tail(check.stdout + check.stderr)}", self.cost
        return ok, f"codex exit {proc.returncode}: {_tail(proc.stdout + proc.stderr)}", self.cost


class RoutingExecutor:
    """Route each task to the right core; verdict/fable tasks return to the human (NeedsHuman)."""

    def __init__(self, *, shell: ShellExecutor, codex: CodexExecutor | None = None) -> None:
        self.shell = shell
        self.codex = codex

    def execute(self, task: dict[str, Any]) -> tuple[bool, str, float]:
        if task["op"] == "verdict" or task["tier"] == "fable":
            raise NeedsHuman(task["id"])                 # judgment is Fable's; hand it back
        if task["tier"] == "codex" and task["op"] == "impl":
            if self.codex is None:
                raise NeedsHuman(task["id"])             # no Codex core reachable here
            return self.codex.execute(task)
        return self.shell.execute(task)                  # test/run/impl on shell-runnable cores
