import json
import re
import unittest

from aioptimizer.context_compiler import ConversationCompiler
from aioptimizer.context_middleware import AttentionContextMiddleware
from experiments.brain_runtime.make_router_fixtures import (
    FIXTURE_CLASSES,
    generate_cases,
    generate_class_cases,
)


VOCABULARY = (
    "cache", "backend", "deployment", "region", "index", "format", "queue",
    "transport", "release", "channel", "memory", "policy", "redis", "sqlite",
    "orion", "vega", "parquet", "arrow", "nats", "kafka", "canary", "stable",
    "newest", "actr",
)


def _embed(texts):
    return [
        [len(re.findall(rf"\b{re.escape(word)}\b", text.lower())) for word in VOCABULARY]
        for text in texts
    ]


def _middleware(*, embed_fn=_embed, budget_chars=1_200):
    return AttentionContextMiddleware(
        budget_chars=budget_chars,
        compiler=ConversationCompiler(embed_fn=embed_fn),
    )


class RouterFixtureTests(unittest.TestCase):
    def test_generators_are_seeded_balanced_and_binding(self):
        first = generate_cases(20260713, 2)
        self.assertEqual(first, generate_cases(20260713, 2))
        self.assertNotEqual(first, generate_cases(20260714, 2))
        self.assertEqual(len(first), 2 * len(FIXTURE_CLASSES))
        self.assertEqual(
            {name: sum(case["fixture_class"] == name for case in first)
             for name in FIXTURE_CLASSES},
            {name: 2 for name in FIXTURE_CLASSES},
        )
        for case in first:
            if case["fixture_class"] == "a":
                self.assertLess(case["history_content_chars"], 12_000)
            elif case["fixture_class"] == "b":
                self.assertGreater(case["history_content_chars"], 12_000)

    def test_frozen_classes_route_and_select_model_free(self):
        middleware = _middleware()
        for case in generate_cases(20260713, 3):
            with self.subTest(case=case["id"]):
                result = middleware.compile_additional_context(
                    case["messages"],
                    query=case["query"],
                    output_budget_chars=case["output_budget_chars"],
                )
                self.assertEqual(result["route"], case["expected_route"])
                if case["expected"]:
                    if case["expected_route"] == "attention":
                        self.assertIn(case["expected"], result["context"].lower())
                for excluded in case["excluded"]:
                    self.assertNotIn(excluded.lower(), result["context"].lower())


class PressureRouterSignalTests(unittest.TestCase):
    def test_stage_one_rejects_repeated_spam_without_encoder(self):
        def forbidden_embed(_texts):
            raise AssertionError("stage two must not run for repeated spam")

        middleware = _middleware(embed_fn=forbidden_embed)
        case = generate_class_cases(20260713, "b", 1)[0]
        result = middleware.compile_additional_context(
            case["messages"],
            query=case["query"],
            output_budget_chars=case["output_budget_chars"],
        )

        self.assertEqual(result["route"], "raw")
        self.assertEqual(result["route_reason"], "covered_by_recent_tail")
        self.assertEqual(result["relevance"]["signal_source"], "repetition_stage")
        self.assertTrue(result["load"]["repetitive"])
        self.assertGreaterEqual(result["load"]["max_token_run"], 24)
        self.assertLess(result["load"]["compression_ratio"], 0.18)
        self.assertEqual(result["embedding_cache_misses"], 0)

    def test_stage_two_reports_old_best_and_recent_tail_coverage(self):
        old_case = generate_class_cases(20260713, "a", 1)[0]
        recent_case = generate_class_cases(20260713, "c", 1)[0]
        middleware = _middleware()

        old = middleware.compile_additional_context(
            old_case["messages"], query=old_case["query"],
            output_budget_chars=old_case["output_budget_chars"],
        )
        recent = middleware.compile_additional_context(
            recent_case["messages"], query=recent_case["query"],
            output_budget_chars=recent_case["output_budget_chars"],
        )

        self.assertEqual(old["route"], "attention")
        self.assertTrue(old["load"]["load_pressure"])
        self.assertGreaterEqual(old["relevance"]["best_record_age_records"], 1)
        self.assertFalse(old["relevance"]["best_in_recent_tail"])
        self.assertEqual(recent["route"], "raw")
        self.assertEqual(recent["route_reason"], "covered_by_recent_tail")
        self.assertTrue(recent["relevance"]["best_in_recent_tail"])

    def test_vague_query_uses_raw_route(self):
        case = generate_class_cases(20260713, "d", 1)[0]
        result = _middleware().compile_additional_context(
            case["messages"], query=case["query"],
            output_budget_chars=case["output_budget_chars"],
        )
        self.assertEqual(result["route"], "raw")
        self.assertEqual(result["route_reason"], "low_relevance")
        self.assertLess(result["relevance"]["peak"], 0.5)

    def test_later_same_topic_decision_wins_and_traceback_is_excluded(self):
        case = generate_class_cases(20260713, "e", 1)[0]
        result = _middleware().compile_additional_context(
            case["messages"], query=case["query"],
            output_budget_chars=case["output_budget_chars"],
        )
        self.assertEqual(result["route"], "attention")
        self.assertIn(case["expected"], result["context"].lower())
        for excluded in case["excluded"]:
            self.assertNotIn(excluded.lower(), result["context"].lower())
        self.assertLess(
            result["relevance"]["rankable_candidates"],
            result["relevance"]["candidates"],
        )

    def test_hard_cost_ceiling_caps_addition(self):
        middleware = _middleware(budget_chars=500)
        case = generate_class_cases(20260713, "a", 1)[0]
        result = middleware.compile_additional_context(
            case["messages"], query=case["query"], output_budget_chars=500,
        )
        self.assertLessEqual(result["output_chars"], 500)
        with self.assertRaisesRegex(ValueError, "between 1 and 500"):
            middleware.compile_additional_context(
                case["messages"], query=case["query"], output_budget_chars=501,
            )


class PressureRouterInvariantTests(unittest.TestCase):
    def test_privacy_filter_runs_before_load_and_semantic_ranking(self):
        secret = "SECRET-9911"

        def privacy_asserting_embed(texts):
            if any(secret in text for text in texts):
                raise AssertionError("private content reached the encoder")
            return _embed(texts)

        messages = [
            {"role": "assistant", "content": "Cache backend decision: use sqlite."},
            {"role": "user", "content": f"{secret} is confidential and must stay hidden."},
            *generate_class_cases(20260713, "a", 1)[0]["messages"][3:],
        ]
        result = _middleware(embed_fn=privacy_asserting_embed).compile_additional_context(
            messages,
            query="What is the cache backend decision?",
            output_budget_chars=450,
        )

        self.assertEqual(result["route"], "attention")
        self.assertIn("sqlite", result["context"].lower())
        self.assertNotIn(secret, result["context"])

    def test_encoder_failure_is_content_free_fail_open(self):
        marker = "sensitive encoder failure marker"

        def failing_embed(_texts):
            raise RuntimeError(marker)

        case = generate_class_cases(20260713, "a", 1)[0]
        result = _middleware(embed_fn=failing_embed).compile_additional_context(
            case["messages"], query=case["query"],
            output_budget_chars=case["output_budget_chars"],
        )

        self.assertEqual(result["route"], "raw")
        self.assertEqual(result["route_reason"], "compiler_failure")
        self.assertTrue(result["fail_open"])
        self.assertEqual(result["context"], "")
        self.assertNotIn(marker, json.dumps(result))


if __name__ == "__main__":
    unittest.main()
