import unittest

from experiments.brain_runtime.session_runtime import PersistentBrainRuntime


class ActrCompactionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.runtime = PersistentBrainRuntime()
        self.old = self.runtime.remember("old", "old", long_term=True)
        self.used = self.runtime.remember("used", "used", long_term=True)
        self.latest = self.runtime.remember("latest", "latest", long_term=True)
        self.runtime.tick(7)
        self.used.uses = 4

    def test_default_actr_policy_keeps_used_memory_and_returns_evicted_count(self):
        evicted = self.runtime.compact_long_term(1)

        self.assertEqual(evicted, 2)
        self.assertEqual(list(self.runtime.long_term_memory), [self.used.id])

    def test_newest_policy_keeps_latest_created_memory(self):
        evicted = self.runtime.compact_long_term(1, policy="newest")

        self.assertEqual(evicted, 2)
        self.assertEqual(list(self.runtime.long_term_memory), [self.latest.id])

    def test_random_policy_is_seeded(self):
        first = self.runtime.compact_long_term(2, policy="random", seed=17)
        kept = list(self.runtime.long_term_memory)

        other = PersistentBrainRuntime()
        for topic in ("old", "used", "latest"):
            other.remember(topic, topic, long_term=True)
        second = other.compact_long_term(2, policy="random", seed=17)

        self.assertEqual(first, 1)
        self.assertEqual(second, 1)
        self.assertEqual(list(other.long_term_memory), kept)

    def test_keep_larger_than_store_evicts_nothing(self):
        self.assertEqual(self.runtime.compact_long_term(10), 0)
        self.assertEqual(len(self.runtime.long_term_memory), 3)

    def test_invalid_keep_raises_value_error_without_changing_store(self):
        before = dict(self.runtime.long_term_memory)

        for keep in (0, -1):
            with self.subTest(keep=keep):
                with self.assertRaises(ValueError):
                    self.runtime.compact_long_term(keep)

        self.assertEqual(self.runtime.long_term_memory, before)

    def test_unknown_policy_raises_value_error_without_changing_store(self):
        before = dict(self.runtime.long_term_memory)

        with self.assertRaises(ValueError):
            self.runtime.compact_long_term(1, policy="missing")

        self.assertEqual(self.runtime.long_term_memory, before)


if __name__ == "__main__":
    unittest.main()
