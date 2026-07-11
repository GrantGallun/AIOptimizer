import unittest

from experiments.brain_runtime import integrated_kernel_eval as ike


class SummarizeMathTests(unittest.TestCase):
    def test_summarize_computes_participation_and_accuracy(self):
        rows = [
            {"first_appearance": True, "correct": False, "incomplete": False,
             "event_kinds": ["retrieve", "reason", "ground"]},
            {"first_appearance": False, "correct": True, "incomplete": False,
             "event_kinds": ["retrieve", "reason", "ground"]},
            {"first_appearance": False, "correct": False, "incomplete": True,
             "event_kinds": []},
        ]
        metrics = {"calls": 6, "malformed_actions": 1}
        summary = ike._summarize("full_kernel", rows, metrics)

        self.assertAlmostEqual(summary["recurrence_accuracy"], 0.5)  # 1 of 2 recurrence rows correct
        self.assertIsInstance(summary["recurrence_ci"], list)
        self.assertEqual(len(summary["recurrence_ci"]), 2)
        self.assertLessEqual(summary["recurrence_ci"][0], summary["recurrence_accuracy"])
        self.assertGreaterEqual(summary["recurrence_ci"][1], summary["recurrence_accuracy"])
        self.assertAlmostEqual(summary["retrieval_participation_rate"], 2 / 3)
        self.assertAlmostEqual(summary["reasoning_participation_rate"], 2 / 3)
        self.assertAlmostEqual(summary["cycle_completion_rate"], 2 / 3)
        self.assertAlmostEqual(summary["malformed_action_rate"], 1 / 6)


class RenderOnlyStructureTests(unittest.TestCase):
    def test_render_only_run_is_structurally_sound(self):
        payload = ike.run_ollama(model="mock", seed=20260711, operator_count=6,
                                 repetitions=2, render_only=True)
        self.assertEqual(set(payload["arms"]), set(ike.ARMS))
        for arm in ike.ARMS:
            for key in ("recurrence_accuracy", "malformed_action_rate", "cycle_completion_rate",
                        "retrieval_participation_rate", "reasoning_participation_rate",
                        "recurrence_ci"):
                self.assertIn(key, payload["arms"][arm])
        # The invariant-bearing arm always retrieves and completes under the mock.
        fk = payload["arms"]["full_kernel"]
        self.assertEqual(fk["cycle_completion_rate"], 1.0)
        self.assertEqual(fk["retrieval_participation_rate"], 1.0)
        self.assertIsInstance(payload["gate"]["observed_recurrence_gap"], float)


if __name__ == "__main__":
    unittest.main()
