import json
import unittest

from gateway.context_compiler import ConversationCompiler
from gateway.context_middleware import AttentionContextMiddleware


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

    def test_small_or_nonchat_requests_pass_through_by_identity(self):
        small = {"messages": self.body["messages"][-2:]}
        self.assertIs(self.middleware.before_request(small), small)
        native = {"model": "qwen3:8b", "prompt": "hello"}
        self.assertIs(self.middleware.before_request(native), native)

    def test_response_is_unchanged(self):
        response = {"choices": [{"message": {"content": "ok"}}]}
        self.assertIs(self.middleware.after_response({}, response), response)

    def test_budget_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            AttentionContextMiddleware(budget_chars=0)


if __name__ == "__main__":
    unittest.main()
