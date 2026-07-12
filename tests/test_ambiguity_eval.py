import unittest

from experiments.brain_runtime.ambiguity_eval import make_cases, score_response, summarize


class AmbiguityEvalTests(unittest.TestCase):
    def test_case_generation_is_deterministic_and_preserves_vague_query(self):
        left = make_cases(17, 3)
        right = make_cases(17, 3)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 3)
        for case in left:
            self.assertEqual(case["messages"][-1]["content"], case["query"])
            self.assertEqual(len(case["signals"]), 4)
            self.assertEqual(set(case["signals"]), set(case["signal_sources"]))

    def test_score_rewards_multiple_grounded_directions_without_fixed_answer(self):
        case = make_cases(19, 1)[0]
        first, second = case["signals"][:2]
        text = (
            f"{first} [{case['signal_sources'][first]}] suggests one path; "
            f"{second} [{case['signal_sources'][second]}] suggests another."
        )
        score = score_response(text, case)
        self.assertEqual(score["direction_count"], 2)
        self.assertEqual(score["evidence_coverage"], 0.5)
        self.assertEqual(score["source_attribution"], 1.0)
        self.assertTrue(score["ambiguity_preserved"])
        self.assertFalse(score["semantic_corruption"])

    def test_score_detects_rigid_goal_invention(self):
        case = make_cases(23, 1)[0]
        score = score_response("The only goal is to must target one niche.", case)
        self.assertFalse(score["ambiguity_preserved"])
        self.assertIn("the only goal", score["rigid_phrases"])

    def test_summary_reports_noninferiority_metrics(self):
        row = {
            "arm": "raw", "direction_count": 3, "evidence_coverage": .75,
            "source_attribution": 1.0, "ambiguity_preserved": True,
            "semantic_corruption": False, "prompt_tokens": 10, "completion_tokens": 20,
            "rewrite_prompt_tokens": 0, "rewrite_completion_tokens": 0,
            "total_duration_ns": 1_000_000_000, "preprocess_seconds": .01,
        }
        result = summarize([row], "raw")
        self.assertEqual(result["mean_direction_count"], 3.0)
        self.assertEqual(result["ambiguity_preservation_rate"], 1.0)
        self.assertEqual(result["model_duration_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
