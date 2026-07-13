import json
import unittest

from aioptimizer.context_compiler import ConversationCompiler
from aioptimizer.context_middleware import AttentionContextMiddleware


def _embed(texts):
    vocabulary = ("cache", "latency", "memory", "privacy")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


class AttentionContextMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.middleware = AttentionContextMiddleware(
            budget_chars=390,
            compiler=ConversationCompiler(embed_fn=_embed),
        )
        self.body = {
            "model": "qwen3:8b",
            "messages": [
                {"role": "system", "content": "Never reveal private memory."},
                {"role": "user", "content": "Discuss unrelated weather " * 8},
                {"role": "assistant", "content": "Weather details " * 8},
                {"role": "user", "content": "Cache latency should stay low."},
                {"role": "assistant", "content": "The exact cache lowers latency."},
                {"role": "user", "content": "What did we decide about cache latency?"},
            ],
        }

    def test_oversized_history_is_organized_but_boundaries_are_verbatim(self):
        out = self.middleware.before_request(self.body)

        self.assertIsNot(out, self.body)
        self.assertEqual(out["messages"][0], self.body["messages"][0])
        self.assertEqual(out["messages"][-1], self.body["messages"][-1])
        self.assertIn("source-grounded", out["messages"][-2]["content"])
        self.assertIn("cache", out["messages"][-2]["content"].lower())
        self.assertLess(len(json.dumps(out)), len(json.dumps(self.body)))
        receipt = self.middleware.receipt_metadata()
        self.assertTrue(receipt["applied"])
        self.assertEqual(receipt["route"], "attention")
        self.assertTrue(receipt["integrity_ok"])
        self.assertTrue(receipt["active_request_preserved"])
        self.assertTrue(receipt["compiler_fingerprint"].startswith("sha256:"))
        self.assertGreater(receipt["embedding_cache_misses"], 0)

    def test_small_or_nonchat_requests_pass_through_by_identity(self):
        small = {"messages": self.body["messages"][-2:]}
        self.assertIs(self.middleware.before_request(small), small)
        native = {"model": "qwen3:8b", "prompt": "hello"}
        self.assertIs(self.middleware.before_request(native), native)
        self.assertFalse(self.middleware.receipt_metadata()["applied"])

    def test_response_is_unchanged(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        self.assertIs(self.middleware.after_response({}, response), response)

    def test_vague_request_bypasses_attention_and_records_reason(self):
        body = json.loads(json.dumps(self.body))
        body["messages"][-1]["content"] = "What stands out?"
        out = self.middleware.before_request(body)

        self.assertIs(out, body)
        receipt = self.middleware.receipt_metadata()
        self.assertFalse(receipt["applied"])
        self.assertEqual(receipt["route"], "raw")
        self.assertEqual(receipt["route_reason"], "low_relevance")
        self.assertLess(receipt["relevance"]["peak"], 0.5)

    def test_budget_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            AttentionContextMiddleware(budget_chars=0)

    def test_additive_compiler_uses_buried_relevance_not_gateway_size(self):
        middleware = AttentionContextMiddleware(
            budget_chars=1_000,
            compiler=ConversationCompiler(embed_fn=_embed),
        )
        history = [
            {"role": "user", "content": "Investigate cache latency."},
            {"role": "assistant", "content": "Decision: exact cache keeps latency low."},
            {"role": "assistant", "content": "Weather discussion " * 12},
            {"role": "user", "content": "Continue with unrelated weather."},
            {"role": "assistant", "content": "Forecast discussion " * 10},
        ]
        focused = middleware.compile_additional_context(
            history,
            query="What did we decide about cache latency?",
            output_budget_chars=250,
        )
        vague = middleware.compile_additional_context(
            history,
            query="What stands out?",
            output_budget_chars=250,
        )

        self.assertLess(focused["history_chars"], middleware.budget_chars)
        self.assertEqual(focused["route"], "attention")
        self.assertIn("cache", focused["context"].lower())
        self.assertLessEqual(focused["output_chars"], 250)
        self.assertEqual(vague["route"], "raw")
        self.assertEqual(vague["route_reason"], "low_relevance")
        self.assertEqual(vague["context"], "")

    def test_additive_compiler_does_not_inject_repeated_recent_spam(self):
        history = [
            *(
                {"role": "assistant", "content": "cache " * 100}
                for _ in range(20)
            ),
            {"role": "user", "content": "Continue."},
        ]

        result = self.middleware.compile_additional_context(
            history,
            query="cache",
            output_budget_chars=300,
        )

        self.assertEqual(result["route"], "raw")
        self.assertEqual(result["route_reason"], "covered_by_recent_tail")
        self.assertGreaterEqual(result["relevance"]["recent_peak"], 0.5)

    def test_additive_compiler_does_not_duplicate_recent_relevant_fact(self):
        history = [
            {"role": "assistant", "content": "Weather history " * 30},
            {"role": "user", "content": "Check cache latency."},
            {"role": "assistant", "content": "The exact cache keeps latency low."},
        ]

        result = self.middleware.compile_additional_context(
            history,
            query="What was the cache latency decision?",
            output_budget_chars=300,
        )

        self.assertEqual(result["route"], "raw")
        self.assertEqual(result["route_reason"], "covered_by_recent_tail")

    def test_additive_compiler_skips_small_history(self):
        result = self.middleware.compile_additional_context(
            [{"role": "user", "content": "cache"}],
            query="cache?",
            output_budget_chars=300,
        )
        self.assertEqual(result["route"], "below_threshold")
        self.assertEqual(result["route_reason"], "history_fits_recent_budget")
        self.assertEqual(result["context"], "")


if __name__ == "__main__":
    unittest.main()
