"""End-to-end pipeline tests: real Board, Cache, WorkspaceClaims, Governor, and a real
ShellExecutor spawning actual OS processes.

Every other agent_bus test exercises one component, usually against SimExecutor. These
run the assembled machine and assert the claims ARCHITECTURE.md makes about it —
parallel execution in separate processes, tier dispatch, capacity, dependency gating,
pre-admission cost control, workspace leases, and (the load-bearing one) out-of-order
execution with in-order retirement. Written 2026-07-17 after an audit found the doc's
"still owed" list was describing a system that no longer existed: nothing ran the whole
pipeline, so nothing caught the drift.

These spawn real subprocesses and take a couple of seconds. That is the point — the
parallelism claim is unfalsifiable without a real clock and real processes.
"""

import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_bus.board import Board
from agent_bus.executors import ExecutorVerifier, ShellExecutor
from agent_bus.scheduler import Scheduler

PY = sys.executable


def sleeper(seconds: float) -> list[str]:
    """A typed argv that occupies a real process for a known wall-clock duration."""
    return [PY, "-c", f"import os,time; print('PID', os.getpid()); time.sleep({seconds})"]


def _pids(board: Board) -> set[str]:
    found = set()
    for task in board.all():
        for token in (task.get("result") or "").split():
            if token.isdigit():
                found.add(token)
    return found


class PipelineParallelismTests(unittest.TestCase):
    def test_tasks_run_concurrently_in_separate_processes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            board = Board(root)
            for i in range(4):
                board.add(op="test", title=f"sleep{i}", tier="sonnet", command=sleeper(1.0))
            sched = Scheduler(root, cores={"sonnet": 4}, executor=ShellExecutor(tmp), budget=1000.0)

            started = time.monotonic()
            events = sched.tick()
            elapsed = time.monotonic() - started

            self.assertEqual(len(events["dispatched"]), 4)
            # Sequential execution would take ~4s. The generous bound keeps this from
            # flaking on a loaded machine while still failing if concurrency is lost.
            self.assertLess(elapsed, 2.5, f"4x1.0s tasks took {elapsed:.2f}s — not concurrent")
            self.assertEqual(len(_pids(board)), 4, "each task must occupy its own OS process")

    def test_capacity_bounds_how_much_load_issues_at_once(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            board = Board(root)
            for i in range(6):
                board.add(op="test", title=f"t{i}", tier="sonnet", command=sleeper(0.05))
            sched = Scheduler(root, cores={"sonnet": 2}, executor=ShellExecutor(tmp), budget=1000.0)
            self.assertEqual(len(sched.tick()["dispatched"]), 2)


class PipelineDispatchTests(unittest.TestCase):
    def test_each_tier_is_claimed_by_a_worker_of_that_tier(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            board = Board(root)
            for tier in ("sonnet", "qwen", "codex"):
                board.add(op="test", title=f"{tier} work", tier=tier, command=sleeper(0.05))
            sched = Scheduler(
                root, cores={"sonnet": 1, "qwen": 1, "codex": 1},
                executor=ShellExecutor(tmp), budget=1000.0,
            )
            sched.tick()
            for task in board.all():
                self.assertTrue(
                    (task.get("owner") or "").startswith(task["tier"]),
                    f"{task['id']} (tier {task['tier']}) claimed by {task.get('owner')}",
                )

    def test_dependent_task_does_not_issue_alongside_its_dependency(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            board = Board(root)
            first = board.add(op="test", title="A", tier="sonnet", command=sleeper(0.05))
            second = board.add(op="test", title="B", tier="sonnet", deps=[first["id"]], command=sleeper(0.05))
            sched = Scheduler(root, cores={"sonnet": 4}, executor=ShellExecutor(tmp), budget=1000.0)
            events = sched.tick()
            self.assertIn(first["id"], events["dispatched"])
            self.assertNotIn(second["id"], events["dispatched"])


class PipelineAdmissionTests(unittest.TestCase):
    def test_governor_refuses_work_that_would_exceed_budget(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            board = Board(root)
            for i in range(3):
                board.add(op="test", title=f"t{i}", tier="sonnet", command=sleeper(0.05))
            sched = Scheduler(root, cores={"sonnet": 3}, executor=ShellExecutor(tmp), budget=0.01)
            events = sched.tick()
            self.assertTrue(events["tripped"])
            self.assertEqual(events["dispatched"], [])

    def test_second_writer_of_a_declared_path_is_deferred_not_raced(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            board = Board(root)
            board.add(op="impl", title="X1", tier="sonnet", writes=["shared.txt"], command=sleeper(0.05))
            board.add(op="impl", title="X2", tier="sonnet", writes=["shared.txt"], command=sleeper(0.05))
            sched = Scheduler(root, cores={"sonnet": 2}, executor=ShellExecutor(tmp), budget=1000.0)
            events = sched.tick()
            self.assertEqual(len(events["dispatched"]), 1)
            self.assertEqual(len(events["deferred"]), 1)


class PipelineReorderBufferTests(unittest.TestCase):
    """ARCHITECTURE.md calls this the load-bearing idea: execute out of order for speed,
    commit in program order so the research record stays consistent."""

    def test_younger_task_finishing_first_still_retires_second(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            board = Board(root)
            # Older task is slow, younger is fast -> execution completes out of seq order.
            slow = board.add(op="test", title="slow-older", tier="sonnet", command=sleeper(1.5))
            fast = board.add(op="test", title="fast-younger", tier="sonnet", command=sleeper(0.05))

            finished: list[str] = []
            retired: list[str] = []

            class OrderRecordingShell(ShellExecutor):
                def execute(self, task):
                    outcome = super().execute(task)
                    finished.append(task["id"])
                    return outcome

            sched = Scheduler(
                root,
                cores={"sonnet": 2},
                executor=OrderRecordingShell(tmp),
                verifier=ExecutorVerifier(ShellExecutor(tmp), reviewer="shell-verifier"),
                budget=1000.0,
                on_retire=lambda task: retired.append(task["id"]),
            )
            sched.run(max_ticks=10)

            self.assertEqual(finished, [fast["id"], slow["id"]], "younger task should finish first")
            self.assertEqual(retired, [slow["id"], fast["id"]], "retirement must follow seq order")
            self.assertTrue(all(t["state"] == "retired" for t in board.all()))


if __name__ == "__main__":
    unittest.main()
