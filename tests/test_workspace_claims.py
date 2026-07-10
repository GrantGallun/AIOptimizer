import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_bus.workspace import WorkspaceClaims, WorkspaceConflict


class WorkspaceClaimsTests(unittest.TestCase):
    def test_multi_path_claim_is_atomic_on_conflict(self):
        with TemporaryDirectory() as tmp:
            claims = WorkspaceClaims(Path(tmp))
            claims.claim(["a.py"], owner="fable")
            with self.assertRaises(WorkspaceConflict):
                claims.claim(["a.py", "b.py"], owner="codex")
            self.assertNotIn("b.py", claims.list()["claims"])

    def test_same_owner_renews_and_owner_checked_release(self):
        now = [100.0]
        with TemporaryDirectory() as tmp:
            claims = WorkspaceClaims(Path(tmp), clock=lambda: now[0])
            first = claims.claim(["src/a.py"], owner="codex", ttl_seconds=10)
            now[0] = 105.0
            second = claims.claim(["src/../src/a.py"], owner="codex", ttl_seconds=20)
            self.assertGreater(second["revision"], first["revision"])
            self.assertEqual(claims.list()["claims"]["src/a.py"]["expires_at"], 125.0)
            with self.assertRaises(WorkspaceConflict):
                claims.release(["src/a.py"], owner="fable")
            self.assertEqual(claims.release(["src/a.py"], owner="codex"), ["src/a.py"])

    def test_expired_claim_is_reclaimed(self):
        now = [10.0]
        with TemporaryDirectory() as tmp:
            claims = WorkspaceClaims(Path(tmp), clock=lambda: now[0])
            claims.claim(["a.py"], owner="fable", ttl_seconds=5)
            now[0] = 16.0
            claims.claim(["a.py"], owner="codex")
            self.assertEqual(claims.list()["claims"]["a.py"]["owner"], "codex")

    def test_rejects_paths_outside_workspace(self):
        with TemporaryDirectory() as tmp:
            claims = WorkspaceClaims(Path(tmp))
            with self.assertRaisesRegex(WorkspaceConflict, "escapes workspace"):
                claims.claim([Path(tmp).parent / "outside.py"], owner="codex")

    def test_concurrent_claim_has_single_winner(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            winners = []
            barrier = threading.Barrier(8)
            lock = threading.Lock()

            def worker(index):
                barrier.wait()
                try:
                    WorkspaceClaims(root).claim(["shared.py"], owner=f"worker-{index}")
                except WorkspaceConflict:
                    return
                with lock:
                    winners.append(index)

            threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(winners), 1)


if __name__ == "__main__":
    unittest.main()
