import unittest

from experiments.brain_runtime.local_worker_eval import context_for, prompt_for, score_response, summarize
from experiments.brain_runtime.multiworker_benchmark import build_cases
from experiments.local_worker.ollama_client import Generation


class LocalWorkerEvaluationTests(unittest.TestCase):
    def test_governed_context_hides_private_worker_claim(self):
        case = build_cases(11)[2]

        notes, sources = context_for(case, "governed_memory")

        self.assertNotIn(case.forbidden, notes)
        self.assertEqual(sources, ["public-test"])

    def test_append_only_context_exposes_private_claim(self):
        case = build_cases(11)[2]

        notes, _ = context_for(case, "append_only")

        self.assertIn(case.forbidden, notes)

    def test_scoring_rejects_private_leak_even_with_expected_value(self):
        case = build_cases(11)[2]

        score = score_response(f"{case.expected} {case.forbidden}", case)

        self.assertFalse(score["success"])
        self.assertTrue(score["privacy_leak"])

    def test_prompt_demands_only_value(self):
        case = build_cases(11)[0]

        prompt = prompt_for(case, "- [test] archive codec is zstd.")

        self.assertIn("no explanation", prompt)
        self.assertIn(case.query, prompt)

    def test_summary_aggregates_local_cost_fields(self):
        row = {
            "policy": "governed_memory",
            "success": True,
            "stale_error": False,
            "privacy_leak": False,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_duration_ns": 2_500_000_000,
        }

        summary = summarize([row], "governed_memory")

        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["prompt_tokens"], 10)
        self.assertEqual(summary["total_duration_seconds"], 2.5)


if __name__ == "__main__":
    unittest.main()
