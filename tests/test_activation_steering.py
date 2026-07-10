import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.activation_steering.steering_harness import (
    DEFAULT_ADVERSARIAL_TASKS,
    DEFAULT_DEV_TASKS,
    DEFAULT_HIDDEN_TASKS,
    DEFAULT_PAIRS,
    DEFAULT_TASKS,
    DEFAULT_TRAIN_PAIRS,
    ToyBehaviorModel,
    compute_toy_vectors,
    parse_number_list,
    parse_string_list,
    read_jsonl,
    run_toy,
)


class ActivationSteeringTests(unittest.TestCase):
    def test_toy_vectors_have_signal(self):
        pairs = read_jsonl(DEFAULT_PAIRS)
        model = ToyBehaviorModel()
        vectors = compute_toy_vectors(pairs, model)

        self.assertIn("evidence", vectors)
        self.assertGreater(vectors["evidence"].values[0], 0)
        self.assertGreater(vectors["clarify"].values[1], 0)
        self.assertGreater(vectors["concise"].values[2], 0)

    def test_toy_steering_improves_success_rate(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "toy_run.json"
            payload = run_toy(
                Namespace(
                    pairs=str(DEFAULT_PAIRS),
                    tasks=str(DEFAULT_TASKS),
                    out=str(out),
                    dims=12,
                    coefficient=1.0,
                )
            )

            self.assertTrue(out.exists())
            summary = payload["summary"]
            self.assertEqual(summary["tasks"], 6)
            self.assertEqual(summary["baseline_success"], 0)
            self.assertEqual(summary["steered_success"], 6)
            self.assertEqual(summary["absolute_success_lift"], 1.0)

    def test_split_files_have_required_fields(self):
        for path in [DEFAULT_TRAIN_PAIRS]:
            rows = read_jsonl(path)
            self.assertGreater(len(rows), 0)
            for row in rows:
                self.assertIn(row["behavior"], {"evidence", "clarify", "concise"})
                self.assertTrue(row["positive"])
                self.assertTrue(row["negative"])

        for path in [DEFAULT_DEV_TASKS, DEFAULT_HIDDEN_TASKS, DEFAULT_ADVERSARIAL_TASKS]:
            rows = read_jsonl(path)
            self.assertGreater(len(rows), 0)
            for row in rows:
                self.assertIn(row["behavior"], {"evidence", "clarify", "concise"})
                self.assertTrue(row["prompt"])
                self.assertTrue(row["positive_answer"])
                self.assertTrue(row["negative_answer"])

    def test_parse_number_list(self):
        self.assertEqual(parse_number_list("1, 2,3", int), [1, 2, 3])
        self.assertEqual(parse_number_list("0.5,2", float), [0.5, 2.0])

    def test_parse_string_list(self):
        self.assertEqual(parse_string_list("raw, orthogonalized"), ["raw", "orthogonalized"])


if __name__ == "__main__":
    unittest.main()
