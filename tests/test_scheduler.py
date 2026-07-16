import subprocess
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_bus.board import Board
from agent_bus.scheduler import COST_UNIT, BranchPredictor, GitCommitter, Governor, Scheduler, SimExecutor
from agent_bus.cache import Cache
from agent_bus.workspace import WorkspaceClaims


class GovernorTests(unittest.TestCase):
    def test_trips_at_budget(self):
        with TemporaryDirectory() as tmp:
            gov = Governor(Cache(Path(tmp)), budget=1.0)
            self.assertFalse(gov.tripped())
            gov.charge(0.6)
            self.assertFalse(gov.tripped())
            gov.charge(0.6)
            self.assertTrue(gov.tripped())

    def test_publishes_seconds_as_the_budget_unit(self):
        with TemporaryDirectory() as tmp:
            cache = Cache(Path(tmp))
            Governor(cache, budget=1.0)
            self.assertEqual(cache.get("governor.unit")["value"], COST_UNIT)


class SchedulerTests(unittest.TestCase):
    class PassVerifier:
        reviewer = "independent-review"

        def verify(self, task, *, producer_ok, producer_result):
            return producer_ok, "independently checked", 0.0

    def _chain(self, board: Board) -> list[str]:
        t1 = board.add(op="impl", title="build", tier="sonnet")
        t2 = board.add(op="test", title="gate", tier="qwen", deps=[t1["id"]])
        t3 = board.add(op="verdict", title="verdict", tier="fable", deps=[t2["id"]])
        return [t1["id"], t2["id"], t3["id"]]

    def test_full_pipeline_retires_in_order(self):
        with TemporaryDirectory() as tmp:
            ids = self._chain(Board(Path(tmp)))
            sched = Scheduler(Path(tmp), budget=100.0)
            sched.run()
            states = {t["id"]: t["state"] for t in Board(Path(tmp)).all()}
            self.assertTrue(all(states[i] == "retired" for i in ids), states)

    def test_budget_ceiling_halts_the_loop(self):
        with TemporaryDirectory() as tmp:
            self._chain(Board(Path(tmp)))
            sched = Scheduler(Path(tmp), budget=0.05)  # below even one sonnet task (0.1)
            history = sched.run()
            self.assertTrue(any(ev["tripped"] for ev in history))
            self.assertEqual([t for t in Board(Path(tmp)).all() if t["state"] == "retired"], [])

    def test_charges_executor_measured_cost_not_tier_estimate(self):
        class MeasuredExecutor:
            def execute(self, task):
                return True, "measured", 2.75

        with TemporaryDirectory() as tmp:
            Board(Path(tmp)).add(op="impl", title="measured work", tier="sonnet")
            sched = Scheduler(Path(tmp), executor=MeasuredExecutor(), budget=10.0)
            sched.run()
            self.assertEqual(sched.governor.spent, 2.75)

    def test_declared_write_conflict_defers_without_execution(self):
        class RecordingExecutor:
            def __init__(self):
                self.calls = []

            def execute(self, task):
                self.calls.append(task["id"])
                return True, "done", 0.1

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            board = Board(root)
            task = board.add(op="impl", title="edit", tier="codex", writes=["src/a.py"])
            external = WorkspaceClaims(root, state_root=root)
            external.claim(["src/a.py"], owner="other-driver")
            executor = RecordingExecutor()
            sched = Scheduler(
                root,
                executor=executor,
                verifier=self.PassVerifier(),
                budget=10.0,
                scheduler_id="test",
            )

            events = sched.tick()
            self.assertEqual(events["deferred"], [task["id"]])
            self.assertEqual(executor.calls, [])
            self.assertEqual(board.get(task["id"])["state"], "ready")

            external.release(["src/a.py"], owner="other-driver")
            sched.run()
            self.assertEqual(executor.calls, [task["id"]])
            self.assertEqual(board.get(task["id"])["state"], "retired")
            self.assertEqual(external.list()["claims"], {})

    def test_real_executor_without_verifier_stays_executed(self):
        class PassingExecutor:
            def execute(self, task):
                return True, "producer pass", 0.1

        with TemporaryDirectory() as tmp:
            task = Board(Path(tmp)).add(op="test", title="needs review", tier="qwen")
            events = Scheduler(Path(tmp), executor=PassingExecutor(), budget=10.0).tick()
            self.assertEqual(events["awaiting_review"], [task["id"]])
            self.assertEqual(Board(Path(tmp)).get(task["id"])["state"], "executed")

    def test_capacity_executes_independent_tasks_concurrently(self):
        class ConcurrentExecutor:
            def __init__(self):
                self.lock = threading.Lock()
                self.barrier = threading.Barrier(2)
                self.active = 0
                self.max_active = 0

            def execute(self, task):
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                self.barrier.wait(timeout=2.0)
                with self.lock:
                    self.active -= 1
                return True, "done", 0.1

        with TemporaryDirectory() as tmp:
            board = Board(Path(tmp))
            first = board.add(op="impl", title="a", tier="sonnet")
            second = board.add(op="impl", title="b", tier="sonnet")
            executor = ConcurrentExecutor()
            events = Scheduler(
                Path(tmp),
                cores={"sonnet": 2},
                executor=executor,
                verifier=self.PassVerifier(),
                budget=10.0,
            ).tick()
            self.assertEqual(executor.max_active, 2)
            self.assertEqual(events["executed"], [first["id"], second["id"]])
            self.assertEqual(events["verified"], [first["id"], second["id"]])

    def test_speculation_commits_on_correct_prediction(self):
        with TemporaryDirectory() as tmp:
            board = Board(Path(tmp))
            gate = board.add(op="test", title="gate", tier="qwen")
            dep = board.add(op="impl", title="speculative-follow-on", tier="sonnet", deps=[gate["id"]])
            # Predict pass (default), gate actually passes -> dependent should be committed + retired.
            Scheduler(Path(tmp), budget=100.0).run()
            states = {t["id"]: t for t in board.all()}
            self.assertFalse(states[dep["id"]]["speculative"])
            self.assertEqual(states[dep["id"]]["state"], "retired")

    def test_speculation_squashes_on_mispredict(self):
        with TemporaryDirectory() as tmp:
            board = Board(Path(tmp))
            gate = board.add(op="test", title="gate", tier="qwen")
            dep = board.add(op="impl", title="speculative-follow-on", tier="sonnet", deps=[gate["id"]])
            # Predict pass, but force the gate to FAIL -> the speculative dependent is squashed.
            sched = Scheduler(Path(tmp), executor=SimExecutor(fail={gate["id"]}), budget=100.0)
            sched.run()
            states = {t["id"]: t for t in board.all()}
            self.assertEqual(states[dep["id"]]["state"], "squashed")
            self.assertNotEqual(states[gate["id"]]["state"], "retired")  # failed gate does not retire

    def test_conservative_predictor_does_not_speculate(self):
        with TemporaryDirectory() as tmp:
            board = Board(Path(tmp))
            gate = board.add(op="test", title="gate", tier="qwen")
            dep = board.add(op="impl", title="follow-on", tier="sonnet", deps=[gate["id"]])
            sched = Scheduler(Path(tmp), budget=100.0, predictor=BranchPredictor(predict_pass=lambda t: False))
            first = sched.tick()  # gate dispatched; with no speculation, dep stays queued this tick
            self.assertNotIn(dep["id"], first["speculated"])


class DurabilityTests(unittest.TestCase):
    def _git(self, repo, *args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)

    def test_retire_hook_fires_per_retired_task(self):
        with TemporaryDirectory() as tmp:
            ids = []
            board = Board(Path(tmp))
            t1 = board.add(op="impl", title="a", tier="sonnet")
            t2 = board.add(op="impl", title="b", tier="sonnet")
            Scheduler(Path(tmp), budget=100.0, on_retire=lambda t: ids.append(t["id"])).run()
            self.assertEqual(ids, [t1["id"], t2["id"]])  # fired in retirement (program) order

    def test_git_committer_commits_a_change_and_skips_clean_tree(self):
        with TemporaryDirectory() as tmp:
            self._git(tmp, "init", "-b", "main")
            self._git(tmp, "config", "user.email", "t@t.t")
            self._git(tmp, "config", "user.name", "t")
            (Path(tmp) / "seed.txt").write_text("x")
            self._git(tmp, "add", "-A")
            self._git(tmp, "commit", "-m", "seed")
            committer = GitCommitter(tmp)
            task = {"id": "t1", "op": "impl", "tier": "sonnet", "title": "noop", "writes": ["improvement.txt"]}
            # No change yet -> no commit.
            self.assertIsNone(committer(task))
            # A real change -> a commit.
            (Path(tmp) / "improvement.txt").write_text("loop output")
            task.update({"id": "t2", "title": "add file"})
            rev = committer(task)
            self.assertTrue(rev)
            log = self._git(tmp, "log", "--oneline").stdout
            self.assertIn("retire t2", log)

    def test_git_committer_preserves_unrelated_staged_and_unstaged_work(self):
        with TemporaryDirectory() as tmp:
            self._git(tmp, "init", "-b", "main")
            self._git(tmp, "config", "user.email", "t@t.t")
            self._git(tmp, "config", "user.name", "t")
            target = Path(tmp) / "target.txt"
            staged_other = Path(tmp) / "staged-other.txt"
            unstaged_other = Path(tmp) / "unstaged-other.txt"
            for path in (target, staged_other, unstaged_other):
                path.write_text("base")
            self._git(tmp, "add", "-A")
            self._git(tmp, "commit", "-m", "seed")

            target.write_text("task change")
            staged_other.write_text("other staged change")
            unstaged_other.write_text("other unstaged change")
            self._git(tmp, "add", "staged-other.txt")
            rev = GitCommitter(tmp)({
                "id": "t1",
                "op": "impl",
                "tier": "codex",
                "title": "scoped edit",
                "writes": ["target.txt"],
            })

            self.assertTrue(rev)
            self.assertEqual(self._git(tmp, "show", "HEAD:target.txt").stdout, "task change")
            self.assertEqual(self._git(tmp, "show", "HEAD:staged-other.txt").stdout, "base")
            self.assertEqual(self._git(tmp, "show", "HEAD:unstaged-other.txt").stdout, "base")
            self.assertEqual(self._git(tmp, "diff", "--cached", "--name-only").stdout.strip(), "staged-other.txt")
            self.assertIn("unstaged-other.txt", self._git(tmp, "diff", "--name-only").stdout)

    def test_git_committer_without_declared_writes_is_noop(self):
        with TemporaryDirectory() as tmp:
            self._git(tmp, "init", "-b", "main")
            self._git(tmp, "config", "user.email", "t@t.t")
            self._git(tmp, "config", "user.name", "t")
            seed = Path(tmp) / "seed.txt"
            seed.write_text("base")
            self._git(tmp, "add", "-A")
            self._git(tmp, "commit", "-m", "seed")
            seed.write_text("changed")
            before = self._git(tmp, "rev-parse", "HEAD").stdout
            task = {"id": "t1", "op": "impl", "tier": "codex", "title": "undeclared"}
            self.assertIsNone(GitCommitter(tmp)(task))
            self.assertEqual(self._git(tmp, "rev-parse", "HEAD").stdout, before)


if __name__ == "__main__":
    unittest.main()
