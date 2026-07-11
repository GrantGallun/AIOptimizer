import unittest

from experiments.brain_runtime.context_selection import make_operators
from experiments.brain_runtime.multihop_eval import ARMS, _observation, build_composed_sequence, run_ollama


class ComposedSequenceTests(unittest.TestCase):
    def setUp(self):
        self.operators = make_operators(6, seed=1)
        self.rows = build_composed_sequence(self.operators, seed=1, n_problems=40)

    def test_expected_is_the_composition(self):
        by_name = {op["name"]: op for op in self.operators}
        for row in self.rows:
            inner_value = by_name[row["inner"]]["fn"](row["a"], row["b"])
            self.assertEqual(row["inner_value"], inner_value)
            self.assertEqual(row["expected"], by_name[row["outer"]]["fn"](inner_value, row["c"]))

    def test_recurrence_requires_both_operators_learned(self):
        seen = set()
        for row in self.rows:
            self.assertEqual(row["first_appearance"], not {row["outer"], row["inner"]} <= seen)
            seen.update({row["outer"], row["inner"]})
        # with 6 operators and 40 problems, most later rows must be recurrences
        self.assertGreater(sum(not r["first_appearance"] for r in self.rows), 25)

    def test_operators_are_distinct_within_a_problem(self):
        for row in self.rows:
            self.assertNotEqual(row["outer"], row["inner"])

    def test_depth_three_expected_is_the_composition(self):
        rows = build_composed_sequence(self.operators, seed=3, n_problems=20, depth=3)
        by_name = {op["name"]: op for op in self.operators}
        for row in rows:
            inner_value = by_name[row["inner"]]["fn"](row["a"], row["b"])
            middle_value = by_name[row["middle"]]["fn"](inner_value, row["c"])
            expected = by_name[row["outer"]]["fn"](middle_value, row["d"])
            self.assertEqual(row["inner_value"], inner_value)
            self.assertEqual(row["middle_value"], middle_value)
            self.assertEqual(row["expected"], expected)
            self.assertEqual(
                _observation(row),
                f"{row['outer']}({row['middle']}({row['inner']}({row['a']}, {row['b']}), "
                f"{row['c']}), {row['d']})",
            )

    def test_depth_three_operators_are_distinct(self):
        rows = build_composed_sequence(self.operators, seed=3, n_problems=20, depth=3)
        for row in rows:
            self.assertEqual(len({row["outer"], row["middle"], row["inner"]}), 3)

    def test_explicit_depth_two_matches_existing_default(self):
        default_rows = build_composed_sequence(self.operators, seed=7, n_problems=10)
        depth_two_rows = build_composed_sequence(self.operators, seed=7, n_problems=10, depth=2)
        self.assertEqual(depth_two_rows, default_rows)

    def test_invalid_depth_raises(self):
        with self.assertRaises(ValueError):
            build_composed_sequence(self.operators, seed=1, depth=4)


class RenderOnlyTests(unittest.TestCase):
    def test_render_only_run_is_structurally_sound(self):
        payload = run_ollama(model="mock", seed=1, operator_count=5, n_problems=12, render_only=True)
        self.assertEqual(set(payload["arms"]), set(ARMS))
        for arm in ARMS:
            summary = payload["arms"][arm]
            self.assertEqual(summary["malformed_action_rate"], 0.0)
            self.assertIn("recurrence_accuracy", summary)
            self.assertIsInstance(summary["recurrence_ci"], list)
            self.assertEqual(len(summary["recurrence_ci"]), 2)
            self.assertLessEqual(summary["recurrence_ci"][0], summary["recurrence_accuracy"])
            self.assertGreaterEqual(summary["recurrence_ci"][1], summary["recurrence_accuracy"])
        # the kernel arm's invariants force completion under the mock
        self.assertEqual(payload["arms"]["full_kernel"]["cycle_completion_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
