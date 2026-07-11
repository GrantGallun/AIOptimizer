import re
import unittest

from experiments.brain_runtime.coala import ActionKind, DecisionContext
from experiments.brain_runtime.coala_learning_eval import (
    ARMS,
    OPERATORS,
    build_sequence,
    evaluate,
    parse_answer,
)


PROBLEM_RE = re.compile(r"Problem: (\w+)\((\d+), (\d+)\)")


def memory_dependent_reasoner(_instruction: str, context: DecisionContext) -> str:
    match = PROBLEM_RE.search(_instruction)
    if match is None or not context.retrieved:
        return "ANSWER=0"
    name, a, b = match.group(1), int(match.group(2)), int(match.group(3))
    expected = OPERATORS[name][0](a, b)
    return f"Applied authorized rule. ANSWER={expected}"


class SequenceTests(unittest.TestCase):
    def test_sequence_is_seeded_balanced_interleaved_and_flagged(self):
        sequence = build_sequence(123)
        self.assertEqual(sequence, build_sequence(123))
        self.assertNotEqual(sequence, build_sequence(124))
        self.assertEqual(len(sequence), 25)
        for name in OPERATORS:
            selected = [row for row in sequence if row["operator"] == name]
            self.assertEqual(len(selected), 5)
            self.assertEqual(sum(row["first_appearance"] for row in selected), 1)
        self.assertTrue(all(2 <= row["a"] <= 9 and 2 <= row["b"] <= 9 for row in sequence))
        for offset in range(0, len(sequence), len(OPERATORS)):
            self.assertEqual({row["operator"] for row in sequence[offset : offset + len(OPERATORS)]}, set(OPERATORS))

    def test_parse_answer_prefers_explicit_marker_then_last_integer(self):
        self.assertEqual(parse_answer("work 2 + 3. ANSWER=8"), 8)
        self.assertEqual(parse_answer("result is 17"), 17)
        self.assertIsNone(parse_answer("unknown"))


class WiringTests(unittest.TestCase):
    def setUp(self):
        self.sequence = build_sequence(123)
        self.payload = evaluate(
            self.sequence,
            reasoners={arm: memory_dependent_reasoner for arm in ARMS},
            model="fake",
            seed=123,
        )

    def test_no_memory_never_retrieves_or_learns(self):
        rows = [row for row in self.payload["rows"] if row["arm"] == "no_memory"]
        self.assertTrue(all(row["event_kinds"] == [ActionKind.REASON.value, ActionKind.GROUND.value] for row in rows))
        self.assertTrue(all(not row["retrieved_memory_ids"] for row in rows))
        self.assertTrue(all(not row["learned_after_feedback"] for row in rows))

    def test_coala_retrieves_before_reason_and_learns_only_after_first_grade(self):
        rows = [row for row in self.payload["rows"] if row["arm"] == "coala"]
        self.assertTrue(all(row["event_kinds"] == [ActionKind.RETRIEVE.value, ActionKind.REASON.value, ActionKind.GROUND.value] for row in rows))
        first = [row for row in rows if row["first_appearance"]]
        recurrence = [row for row in rows if not row["first_appearance"]]
        self.assertTrue(all(not row["retrieved_memory_ids"] for row in first))
        self.assertTrue(all(row["learned_after_feedback"] for row in first))
        self.assertTrue(all(len(row["retrieved_memory_ids"]) == 1 for row in recurrence))
        self.assertTrue(all(not row["learned_after_feedback"] for row in recurrence))

    def test_model_free_control_does_not_learn_and_coala_recurrence_does(self):
        summaries = {row["arm"]: row for row in self.payload["summaries"]}
        self.assertEqual(summaries["no_memory"]["first_appearance_accuracy"], 0.0)
        self.assertEqual(summaries["no_memory"]["recurrence_accuracy"], 0.0)
        self.assertEqual(summaries["no_memory"]["learned_memories"], 0)
        self.assertEqual(summaries["coala"]["first_appearance_accuracy"], 0.0)
        self.assertEqual(summaries["coala"]["recurrence_accuracy"], 1.0)
        self.assertEqual(summaries["coala"]["learned_memories"], 5)

    def test_output_contains_fable_owned_gate_components_without_verdict(self):
        gate = self.payload["gate"]
        self.assertEqual(gate["owner"], "fable")
        self.assertEqual(gate["required_recurrence_accuracy_gap"], 0.30)
        self.assertEqual(gate["observed_recurrence_accuracy_gap"], 1.0)
        self.assertNotIn("verdict", gate)


if __name__ == "__main__":
    unittest.main()
