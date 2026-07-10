import unittest

from experiments.brain_runtime.context_selection import (
    build_prompt,
    make_operators,
    parse_int,
    select_topk,
)


class ContextSelectionTests(unittest.TestCase):
    def test_make_operators_unique_names_and_consistent_fn(self):
        operators = make_operators(30, seed=1)

        self.assertEqual(len(operators), 30)
        names = [op["name"] for op in operators]
        self.assertEqual(len(names), len(set(names)))

        for op in operators:
            result_a = op["fn"](3, 4)
            result_b = op["fn"](3, 4)
            self.assertIsInstance(result_a, int)
            self.assertEqual(result_a, result_b)

    def test_parse_int(self):
        self.assertEqual(parse_int("ans: -7"), -7)
        self.assertIsNone(parse_int("none"))

    def test_build_prompt_contains_rules_and_call(self):
        prompt = build_prompt("glorp", 4, 7, ["glorp(a,b) = a + b + 3"])

        self.assertIn("Rules you must apply", prompt)
        self.assertIn("glorp(4, 7)", prompt)

    def test_select_topk_includes_target_rule(self):
        target = "glorp(a,b) = a + b + 3"
        distractors = [f"zibaz{i}(a,b) = a + b + {i}" for i in range(10)]
        try:
            selected = select_topk("glorp", [target] + distractors, 3)
        except Exception:
            self.skipTest("Encoder unavailable in this environment.")

        self.assertEqual(len(selected), 3)
        self.assertIn(target, selected)


if __name__ == "__main__":
    unittest.main()
