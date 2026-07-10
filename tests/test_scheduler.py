import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_bus.board import Board
from agent_bus.scheduler import BranchPredictor, GitCommitter, Governor, Scheduler, SimExecutor
from agent_bus.cache import Cache


class GovernorTests(unittest.TestCase):
    def test_trips_at_budget(self):
        with TemporaryDirectory() as tmp:
            gov = Governor(Cache(Path(tmp)), budget=1.0)
            self.assertFalse(gov.tripped())
            gov.charge(0.6)
            self.assertFalse(gov.tripped())
            gov.charge(0.6)
            self.assertTrue(gov.tripped())


class SchedulerTests(unittest.TestCase):
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
            # No change yet -> no commit.
            self.assertIsNone(committer({"id": "t1", "op": "impl", "tier": "sonnet", "title": "noop"}))
            # A real change -> a commit.
            (Path(tmp) / "improvement.txt").write_text("loop output")
            rev = committer({"id": "t2", "op": "impl", "tier": "sonnet", "title": "add file"})
            self.assertTrue(rev)
            log = self._git(tmp, "log", "--oneline").stdout
            self.assertIn("retire t2", log)


if __name__ == "__main__":
    unittest.main()
