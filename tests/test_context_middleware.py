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

    def test_additive_compiler_injects_only_for_large_focused_history(self):
        history = self.body["messages"][1:-1]
        focused = self.middleware.compile_additional_context(
            history,
            query="What did we decide about cache latency?",
            output_budget_chars=300,
        )
        vague = self.middleware.compile_additional_context(
            history,
            query="What stands out?",
            output_budget_chars=300,
        )

        self.assertEqual(focused["route"], "attention")
        self.assertIn("cache", focused["context"].lower())
        self.assertLessEqual(focused["output_chars"], 300)
        self.assertEqual(vague["route"], "raw")
        self.assertEqual(vague["context"], "")

    def test_additive_compiler_skips_small_history(self):
        result = self.middleware.compile_additional_context(
            [{"role": "user", "content": "cache"}],
            query="cache?",
            output_budget_chars=300,
        )
        self.assertEqual(result["route"], "below_threshold")
        self.assertEqual(result["context"], "")


if __name__ == "__main__":
    unittest.main()
