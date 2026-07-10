import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from agent_bus.board import Board
from agent_bus.executors import CommandPolicy, CommandRejected, CodexExecutor, NeedsHuman, RoutingExecutor, ShellExecutor
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
        ex = ShellExecutor(cwd=".")
        ok, result, cost = ex.execute({"acceptance": f'"{sys.executable}" -c "print(1)"'})
        self.assertTrue(ok, result)
        self.assertGreater(cost, 0)

    def test_nonzero_exit_is_failure(self):
        ex = ShellExecutor(cwd=".")
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
        ex = ShellExecutor(cwd=".", allowlist=['"' + sys.executable])
        ok, result, _ = ex.execute({"acceptance": f'"{sys.executable}" -c "print(1)"'})
        self.assertTrue(ok, result)

    def test_no_allowlist_backward_compatible(self):
        ex = ShellExecutor(cwd=".")
        ok, result, _ = ex.execute({"acceptance": f'"{sys.executable}" -c "print(1)"'})
        self.assertTrue(ok, result)

    def test_cost_is_measured_elapsed_seconds(self):
        ticks = iter([10.0, 12.5])
        ex = ShellExecutor(cwd=".", clock=lambda: next(ticks))
        completed = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        with patch("agent_bus.executors.subprocess.run", return_value=completed):
            ok, _, cost = ex.execute({"acceptance": "python -m unittest"})
        self.assertTrue(ok)
        self.assertEqual(cost, 2.5)


class CodexExecutorCostTests(unittest.TestCase):
    def test_cost_includes_codex_and_acceptance_elapsed_seconds(self):
        ticks = iter([20.0, 23.75])
        ex = CodexExecutor(cwd=".", clock=lambda: next(ticks))
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
        return RoutingExecutor(shell=ShellExecutor(cwd="."), codex=None)

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
            sched = Scheduler(Path(tmp), executor=RoutingExecutor(shell=shell, codex=None), budget=100.0)
            sched.run()
            states = {t["id"]: t["state"] for t in board.all()}
            self.assertEqual(states[work["id"]], "retired")     # real shell work completed + committed
            self.assertEqual(states[verdict["id"]], "blocked")  # verdict handed back to the human


if __name__ == "__main__":
    unittest.main()
