import random
import unittest
from dataclasses import dataclass

from experiments.brain_runtime.compaction_policies import (
    keep_actr,
    keep_newest,
    keep_random,
)


@dataclass
class Item:
    id: str
    created_at: int
    last_accessed: int
    uses: int


@dataclass
class ItemWithoutCreatedAt:
    id: str
    last_accessed: int = 0
    uses: int = 0


class CompactionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            Item("old", created_at=2, last_accessed=5, uses=0),
            Item("newest", created_at=9, last_accessed=9, uses=0),
            Item("middle", created_at=5, last_accessed=6, uses=1),
            Item("popular", created_at=1, last_accessed=3, uses=4),
        ]

    def test_newest_selects_by_created_at(self):
        kept = keep_newest(self.items, 2, clock=10)

        self.assertEqual([item.id for item in kept], ["newest", "middle"])

    def test_newest_falls_back_to_insertion_order(self):
        items = [ItemWithoutCreatedAt("a"), ItemWithoutCreatedAt("b"), ItemWithoutCreatedAt("c")]

        kept = keep_newest(items, 2, clock=10)

        self.assertEqual([item.id for item in kept], ["c", "b"])

    def test_random_matches_seeded_sample_and_is_deterministic(self):
        expected = random.Random(73).sample(self.items, k=2)

        first = keep_random(self.items, 2, clock=10, seed=73)
        second = keep_random(self.items, 2, clock=999, seed=73)

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_actr_selects_by_frozen_activation_score(self):
        kept = keep_actr(self.items, 2, clock=10)

        self.assertEqual([item.id for item in kept], ["popular", "middle"])

    def test_actr_breaks_score_ties_by_id(self):
        items = [
            Item("z", created_at=1, last_accessed=8, uses=2),
            Item("a", created_at=2, last_accessed=8, uses=2),
        ]

        kept = keep_actr(items, 2, clock=10)

        self.assertEqual([item.id for item in kept], ["a", "z"])

    def test_keep_larger_than_input_keeps_every_item(self):
        for policy in (keep_newest, keep_random, keep_actr):
            with self.subTest(policy=policy.__name__):
                kept = policy(self.items, 10, clock=10, seed=19)
                self.assertCountEqual(kept, self.items)

    def test_non_positive_keep_raises_value_error(self):
        for policy in (keep_newest, keep_random, keep_actr):
            for keep in (0, -1):
                with self.subTest(policy=policy.__name__, keep=keep):
                    with self.assertRaises(ValueError):
                        policy(self.items, keep, clock=10)


if __name__ == "__main__":
    unittest.main()
