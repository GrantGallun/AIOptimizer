import unittest

from experiments.brain_runtime.reasoning_memory import (
    OPERATORS,
    build_prompt,
    build_sequence,
    parse_int,
)
from experiments.brain_runtime.runtime import BrainRuntime


class ReasoningMemorySequenceTests(unittest.TestCase):
    def test_sequence_has_25_problems_with_five_reps_per_operator(self):
        sequence = build_sequence(101)

        self.assertEqual(len(sequence), 25)
        counts = {name: 0 for name in OPERATORS}
        for problem in sequence:
            counts[problem["op"]] += 1
        for name in OPERATORS:
            self.assertEqual(counts[name], 5)

    def test_exactly_five_first_appearances(self):
        sequence = build_sequence(101)

        first_appearances = [problem for problem in sequence if problem["first_appearance"]]
        self.assertEqual(len(first_appearances), 5)
        self.assertEqual({problem["op"] for problem in first_appearances}, set(OPERATORS))

    def test_expected_matches_operator_function(self):
        sequence = build_sequence(101)

        for problem in sequence:
            fn, _rule_text = OPERATORS[problem["op"]]
            self.assertEqual(problem["expected"], fn(problem["a"], problem["b"]))


class ParseIntTests(unittest.TestCase):
    def test_parses_negative_integer_from_sentence(self):
        self.assertEqual(parse_int("The answer is -4."), -4)

    def test_returns_none_when_no_number_present(self):
        self.assertIsNone(parse_int("no number"))


class BrainRuntimeRoundTripTests(unittest.TestCase):
    def test_remember_then_retrieve_returns_stored_rule(self):
        runtime = BrainRuntime()
        runtime.remember(topic="glorp", content="glorp(a,b) = a + b + 3", key="glorp")

        results = runtime.retrieve("glorp", limit=2)

        self.assertTrue(results)
        self.assertIn("glorp", results[0].content)


class BuildPromptTests(unittest.TestCase):
    def test_prompt_includes_known_rules_when_lessons_present(self):
        problem = {"op": "glorp", "a": 2, "b": 3}
        prompt = build_prompt(problem, ["glorp(a,b) = a + b + 3"])

        self.assertIn("Rules you must apply", prompt)
        self.assertIn("glorp(a,b) = a + b + 3", prompt)

    def test_prompt_omits_known_rules_when_no_lessons(self):
        problem = {"op": "glorp", "a": 2, "b": 3}
        prompt = build_prompt(problem, [])

        self.assertNotIn("Rules you must apply", prompt)


if __name__ == "__main__":
    unittest.main()
