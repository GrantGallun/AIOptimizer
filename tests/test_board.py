import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_bus.board import Board, BoardConflict


class ScoreboardTests(unittest.TestCase):
    def test_dependency_gates_readiness(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            t1 = b.add(op="impl", title="build", tier="sonnet")
            t2 = b.add(op="test", title="verify", tier="sonnet", deps=[t1["id"]])
            # t1 has no deps -> ready; t2 blocked until t1 verified.
            self.assertIsNotNone(b.dispatch(tier="sonnet", worker="s1"))     # gets t1
            self.assertIsNone(b.dispatch(tier="sonnet", worker="s2"))        # t2 not ready

    def test_declared_write_set_is_deduplicated_and_persisted(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            task = b.add(
                op="impl",
                title="edit parser",
                tier="codex",
                writes=["src/parser.py", "src/parser.py", "tests/test_parser.py"],
            )
            self.assertEqual(task["writes"], ["src/parser.py", "tests/test_parser.py"])
            self.assertEqual(b.get(task["id"])["writes"], task["writes"])

    def test_typed_command_is_validated_and_persisted(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            task = b.add(op="test", title="typed", tier="qwen", command=["python", "-m", "unittest"])
            self.assertEqual(b.get(task["id"])["command"], ["python", "-m", "unittest"])
            with self.assertRaises(ValueError):
                b.add(op="test", title="empty", tier="qwen", command=[])

    def test_defer_requires_current_owner_and_makes_task_ready(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            task = b.add(op="impl", title="edit", tier="codex")
            dispatched = b.dispatch(tier="codex", worker="codex-a")
            with self.assertRaises(BoardConflict):
                b.defer(task["id"], worker="codex-b")
            deferred = b.defer(task["id"], worker=dispatched["owner"], reason="path busy")
            self.assertEqual(deferred["state"], "ready")
            self.assertIsNone(deferred["owner"])
            self.assertIn("path busy", deferred["result"])

    def test_out_of_order_issue_skips_blocked_elder(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            blocker = b.add(op="spec", title="spec", tier="fable")
            elder = b.add(op="impl", title="elder-blocked", tier="sonnet", deps=[blocker["id"]])
            younger = b.add(op="impl", title="younger-ready", tier="sonnet")
            got = b.dispatch(tier="sonnet", worker="s1")
            self.assertEqual(got["id"], younger["id"])  # younger ready task issues before blocked elder

    def test_review_requires_different_core(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            t = b.add(op="impl", title="x", tier="sonnet")
            b.dispatch(tier="sonnet", worker="sonnet-a")
            b.submit(t["id"], worker="sonnet-a", result="done")
            with self.assertRaises(BoardConflict):
                b.review(t["id"], reviewer="sonnet-a", ok=True)   # self-review blocked
            self.assertEqual(b.review(t["id"], reviewer="codex", ok=True)["state"], "verified")

    def test_rejected_task_can_retry_without_losing_attempt_evidence(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            task = b.add(op="impl", title="repairable", tier="codex")
            dispatched = b.dispatch(tier="codex", worker="codex-a")
            b.submit(task["id"], worker=dispatched["owner"], result="producer output")
            rejected = b.review(task["id"], reviewer="shell-review", ok=False,
                                note="acceptance failed")
            self.assertEqual(rejected["state"], "failed")

            retry = b.retry(task["id"], by="scheduler", feedback="fix test_x")
            self.assertEqual(retry["state"], "ready")
            self.assertEqual(retry["attempt"], 1)
            self.assertEqual(retry["retry_feedback"], "fix test_x")
            self.assertIsNone(retry["owner"])
            self.assertIsNone(retry["reviewer"])
            self.assertEqual(len(retry["attempt_history"]), 1)
            self.assertIn("acceptance failed", retry["attempt_history"][0]["result"])
            self.assertEqual(
                b.dispatch(tier="codex", worker="codex-b")["id"], task["id"]
            )

    def test_retry_rejects_nonfailed_task(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            task = b.add(op="impl", title="not rejected", tier="codex")
            with self.assertRaises(BoardConflict):
                b.retry(task["id"], by="scheduler")

    def test_retirement_is_in_order_and_fable_only(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            t1 = b.add(op="impl", title="first", tier="sonnet")
            t2 = b.add(op="impl", title="second", tier="sonnet")
            # Dispatch both, then execute + verify both (out of order is fine).
            for _ in range(2):
                task = b.dispatch(tier="sonnet", worker=f"s-{_}")
                b.submit(task["id"], worker=task["owner"], result="r")
                b.review(task["id"], reviewer="codex", ok=True)
            with self.assertRaises(BoardConflict):
                b.retire(t2["id"], by="fable")          # can't retire seq 2 before seq 1
            with self.assertRaises(BoardConflict):
                b.retire(t1["id"], by="sonnet")         # only Fable retires
            self.assertEqual(b.retire(t1["id"], by="fable")["state"], "retired")
            self.assertEqual(b.retire(t2["id"], by="fable")["state"], "retired")

    def test_watchdog_reclaims_dispatched(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            t = b.add(op="impl", title="x", tier="sonnet")
            dispatched = b.dispatch(tier="sonnet", worker="ghost", lease_seconds=10)
            self.assertEqual(b.watchdog(now=dispatched["lease_expires_at"] - 1), [])
            self.assertEqual(b.watchdog(now=dispatched["lease_expires_at"]), [t["id"]])
            # reclaimed -> dispatchable again
            self.assertIsNotNone(b.dispatch(tier="sonnet", worker="s2"))

    def test_concurrent_add_no_lost_seq(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))

            def worker():
                for _ in range(10):
                    b.add(op="impl", title="x", tier="sonnet")

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            tasks = b.all()
            self.assertEqual(len(tasks), 40)
            ids = {t["id"] for t in tasks}
            self.assertEqual(ids, {f"t{i:04d}" for i in range(1, 41)})

    def test_concurrent_dispatch_no_double_claim(self):
        with TemporaryDirectory() as tmp:
            b = Board(Path(tmp))
            expected_ids = {b.add(op="impl", title="x", tier="sonnet")["id"] for _ in range(20)}

            collected: list[str] = []
            lock = threading.Lock()

            def worker(name):
                while True:
                    task = b.dispatch(tier="sonnet", worker=name)
                    if task is None:
                        break
                    with lock:
                        collected.append(task["id"])

            threads = [threading.Thread(target=worker, args=(f"w{n}",)) for n in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(collected), 20)
            self.assertEqual(set(collected), expected_ids)
            self.assertEqual(len(collected), len(set(collected)))  # no duplicates
            for task_id in expected_ids:
                self.assertIsNotNone(b.get(task_id)["owner"])


if __name__ == "__main__":
    unittest.main()
