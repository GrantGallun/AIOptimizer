import unittest

from experiments.brain_runtime.coala import ActionKind, DecisionContext
from experiments.brain_runtime.coala_learning_eval_scale import (
    RETRIEVAL_MODES,
    EncoderMemoryRuntime,
    build_sequence,
    evaluate,
)
from experiments.brain_runtime.context_selection import make_operators


def memory_dependent_reasoner(instruction: str, context: DecisionContext) -> str:
    if not context.retrieved:
        return "ANSWER=0"
    problem = instruction.rsplit("Problem: ", 1)[-1]
    name, args = problem.split("(", 1)
    a, b = (int(value) for value in args.rstrip(")").split(","))
    # The test intentionally relies on the authorized memory's operator key
    # only for selection; the expected arithmetic comes from the generated rule
    # function in the test's sequence construction.
    values = {row["operator"]: row for row in TEST_SEQUENCE}
    operator = next(item for item in TEST_OPERATORS if item["name"] == name)
    return f"ANSWER={operator['fn'](a, b)}" if values[name] else "ANSWER=0"


TEST_OPERATORS = make_operators(6, seed=9)
TEST_SEQUENCE = build_sequence(TEST_OPERATORS, seed=10, repetitions=3)


class ScaleHarnessTests(unittest.TestCase):
    def test_sequence_is_balanced_and_flags_one_first_appearance(self):
        self.assertEqual(len(TEST_SEQUENCE), 18)
        for operator in TEST_OPERATORS:
            rows = [row for row in TEST_SEQUENCE if row["operator"] == operator["name"]]
            self.assertEqual(len(rows), 3)
            self.assertEqual(sum(row["first_appearance"] for row in rows), 1)

    def test_encoder_runtime_uses_injected_selector_and_returns_selected_memories(self):
        calls = []

        def selector(query, candidates, limit):
            calls.append((query, candidates, limit))
            return candidates[-limit:]

        runtime = EncoderMemoryRuntime(selector=selector)
        runtime.remember(
            "operator:target", "Verified operator rule: target(a,b) = a + b + 1",
            key="operator:target", value="target(a,b) = a + b + 1", long_term=True,
        )
        other = runtime.remember(
            "operator:other", "Verified operator rule: other(a,b) = a + b + 2",
            key="operator:other", value="other(a,b) = a + b + 2", long_term=True,
        )
        selected = runtime.retrieve("operator:target", limit=1)
        self.assertEqual(selected, [other])
        self.assertEqual(calls[0][0], "operator:target")
        self.assertEqual(calls[0][2], 1)

    def test_model_free_modes_keep_coala_cycle_and_learn_after_first_grade(self):
        payload = evaluate(
            TEST_SEQUENCE,
            reasoners={mode: memory_dependent_reasoner for mode in RETRIEVAL_MODES},
            model="fake",
            seed=10,
            retrieval_limit=2,
        )
        self.assertEqual({row["mode"] for row in payload["rows"]}, set(RETRIEVAL_MODES))
        for mode in RETRIEVAL_MODES:
            rows = [row for row in payload["rows"] if row["mode"] == mode]
            self.assertTrue(all(row["event_kinds"] == [ActionKind.RETRIEVE.value, ActionKind.REASON.value, ActionKind.GROUND.value] for row in rows))
            self.assertTrue(all(row["learned_after_feedback"] == row["first_appearance"] for row in rows))
            summary = next(item for item in payload["summaries"] if item["mode"] == mode)
            self.assertEqual(summary["learned_memories"], len(TEST_OPERATORS))

    def test_gate_is_fable_owned_and_contains_encoder_jaccard_gap(self):
        payload = evaluate(
            TEST_SEQUENCE,
            reasoners={mode: memory_dependent_reasoner for mode in RETRIEVAL_MODES},
            model="fake",
            seed=10,
        )
        self.assertEqual(payload["gate"]["owner"], "fable")
        self.assertEqual(payload["gate"]["required_recurrence_accuracy_gap"], 0.20)
        self.assertIn("observed_recurrence_accuracy_gap", payload["gate"])
        self.assertNotIn("verdict", payload["gate"])


if __name__ == "__main__":
    unittest.main()
