"""Unit tests for the v14 gate runner's pure functions (no encoder, no middleware)."""

import unittest

from experiments.brain_runtime.make_router_fixtures import generate_class_cases
from experiments.brain_runtime.run_v14_gate import (
    PLANTED_SECRET,
    privacy_case,
    score_case,
    templated_variant,
)


class TemplatedVariantTests(unittest.TestCase):
    def test_variant_keeps_target_and_query_but_replaces_chatter(self):
        case = generate_class_cases(20260713, "a", 1)[0]
        variant = templated_variant(case)
        self.assertEqual(variant["fixture_class"], "a_templated")
        self.assertEqual(variant["messages"][:2], case["messages"][:2])
        self.assertEqual(len(variant["messages"]), len(case["messages"]))
        self.assertEqual(variant["query"], case["query"])
        self.assertEqual(variant["expected"], case["expected"])
        chatter = " ".join(m["content"] for m in variant["messages"][2:])
        self.assertNotIn("Worklog", chatter)

    def test_variant_is_deterministic_and_rejects_other_classes(self):
        case = generate_class_cases(20260713, "a", 1)[0]
        self.assertEqual(templated_variant(case), templated_variant(case))
        spam = generate_class_cases(20260713, "b", 1)[0]
        with self.assertRaises(ValueError):
            templated_variant(spam)


class ScoreCaseTests(unittest.TestCase):
    def test_class_a_requires_attention_route_with_expected_in_context(self):
        case = generate_class_cases(20260713, "a", 1)[0]
        hit = {"route": "attention", "context": f"...{case['expected']}...", "route_reason": None}
        miss_route = {"route": "raw", "context": "", "route_reason": "covered_by_recent_tail"}
        miss_content = {"route": "attention", "context": "unrelated", "route_reason": None}
        self.assertTrue(score_case(case, hit)["correct"])
        self.assertFalse(score_case(case, miss_route)["correct"])
        self.assertFalse(score_case(case, miss_content)["correct"])

    def test_class_b_requires_raw_without_encoder_spend(self):
        case = generate_class_cases(20260713, "b", 1)[0]
        stage_one = {"route": "raw", "context": "", "embedding_cache_misses": 0}
        paid_encoder = {"route": "raw", "context": "", "embedding_cache_misses": 3}
        self.assertTrue(score_case(case, stage_one)["correct"])
        self.assertFalse(score_case(case, paid_encoder)["correct"])

    def test_class_e_requires_exclusions_absent(self):
        case = generate_class_cases(20260713, "e", 1)[0]
        clean = {"route": "attention", "context": f"use {case['expected']} now"}
        leaked = {
            "route": "attention",
            "context": f"use {case['expected']} now\nTraceback (most recent call last)",
        }
        self.assertTrue(score_case(case, clean)["correct"])
        self.assertFalse(score_case(case, leaked)["correct"])

    def test_classes_c_and_d_check_reasons(self):
        c_case = generate_class_cases(20260713, "c", 1)[0]
        d_case = generate_class_cases(20260713, "d", 1)[0]
        self.assertTrue(score_case(c_case, {"route": "raw", "route_reason": "covered_by_recent_tail"})["correct"])
        self.assertFalse(score_case(c_case, {"route": "raw", "route_reason": "low_relevance"})["correct"])
        self.assertTrue(score_case(d_case, {"route": "raw", "route_reason": "low_relevance"})["correct"])
        self.assertFalse(score_case(d_case, {"route": "attention", "route_reason": None, "context": "x"})["correct"])


class PrivacyCaseTests(unittest.TestCase):
    def test_secret_is_planted_mid_history_without_mutating_the_original(self):
        case = generate_class_cases(20260713, "a", 1)[0]
        before = [dict(m) for m in case["messages"]]
        planted = privacy_case(case)
        self.assertEqual(case["messages"], before)
        self.assertEqual(len(planted["messages"]), len(before) + 1)
        self.assertTrue(any(PLANTED_SECRET in m["content"] for m in planted["messages"]))


if __name__ == "__main__":
    unittest.main()
