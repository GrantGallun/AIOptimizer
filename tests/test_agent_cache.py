import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_bus.cache import Cache, CacheConflict


class SharedCacheTests(unittest.TestCase):
    def test_set_versions_and_global_rev_increment(self):
        with TemporaryDirectory() as tmp:
            c = Cache(Path(tmp))
            e1 = c.set("task", "v2.1", writer="fable")
            e2 = c.set("task", "v2.1 in progress", writer="codex")
            self.assertEqual((e1["version"], e2["version"]), (1, 2))
            self.assertGreater(e2["rev"], e1["rev"])
            self.assertEqual(c.get("task")["value"], "v2.1 in progress")

    def test_compare_and_set_detects_stale_write(self):
        with TemporaryDirectory() as tmp:
            c = Cache(Path(tmp))
            c.set("gate", "open", writer="fable")            # version 1
            c.set("gate", "closed", writer="codex")          # version 2, unseen by the next writer
            with self.assertRaises(CacheConflict):
                c.set("gate", "open", writer="fable", expect_version=1)  # stale CAS -> conflict
            # A correct CAS against the current version succeeds.
            self.assertEqual(c.set("gate", "reopened", writer="fable", expect_version=2)["version"], 3)

    def test_claim_blocks_other_writer_until_release(self):
        with TemporaryDirectory() as tmp:
            c = Cache(Path(tmp))
            c.claim("harness", writer="codex")
            with self.assertRaises(CacheConflict):
                c.set("harness", "fable edit", writer="fable")   # line owned by codex
            self.assertTrue(c.set("harness", "codex edit", writer="codex")["version"] >= 1)  # owner can write
            c.release("harness", writer="codex")
            self.assertEqual(c.set("harness", "fable edit", writer="fable")["writer"], "fable")

    def test_pull_since_returns_only_newer_entries(self):
        with TemporaryDirectory() as tmp:
            c = Cache(Path(tmp))
            c.set("a", "1", writer="fable")
            snapshot = c.pull(since_rev=0)["rev"]
            c.set("b", "2", writer="codex")
            delta = c.pull(since_rev=snapshot)
            changed_keys = [e["key"] for e in delta["changed"]]
            self.assertEqual(changed_keys, ["b"])
            self.assertGreater(delta["rev"], snapshot)

    def test_concurrent_distinct_key_writes_do_not_lose_updates(self):
        # The race Codex flagged: without a cross-process lock around load-modify-save,
        # concurrent writers to different keys clobber each other (each loads the same
        # base state and last-writer-wins overwrites the whole file). With the lock every
        # write lands. 4 writers x 10 distinct keys => 40 keys and rev == 40.
        with TemporaryDirectory() as tmp:
            cache = Cache(Path(tmp))
            writers, per = 4, 10

            def work(w):
                for i in range(per):
                    cache.set(f"k.{w}.{i}", "v", writer="fable")

            threads = [threading.Thread(target=work, args=(w,)) for w in range(writers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            state = cache._load()
            self.assertEqual(len(state["entries"]), writers * per)  # no lost keys
            self.assertEqual(state["rev"], writers * per)           # no lost rev increments

    def test_render_writes_dashboard(self):
        with TemporaryDirectory() as tmp:
            c = Cache(Path(tmp))
            c.set("status.fable", "orchestrating", writer="fable")
            text = c.shared.read_text(encoding="utf-8")
            self.assertIn("status.fable", text)
            self.assertIn("Shared Cache", text)


if __name__ == "__main__":
    unittest.main()
