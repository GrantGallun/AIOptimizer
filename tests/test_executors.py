import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_bus.board import Board
from agent_bus.executors import CodexExecutor, NeedsHuman, RoutingExecutor, ShellExecutor
from agent_bus.scheduler import Scheduler


class ShellExecutorTests(unittest.TestCase):
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
                             acceptance=f'"{sys.executable}" -c "print(1)"')
            verdict = board.add(op="verdict", title="human call", tier="fable", deps=[work["id"]])
            sched = Scheduler(Path(tmp), executor=RoutingExecutor(shell=ShellExecutor(cwd="."), codex=None), budget=100.0)
            sched.run()
            states = {t["id"]: t["state"] for t in board.all()}
            self.assertEqual(states[work["id"]], "retired")     # real shell work completed + committed
            self.assertEqual(states[verdict["id"]], "blocked")  # verdict handed back to the human


if __name__ == "__main__":
    unittest.main()
