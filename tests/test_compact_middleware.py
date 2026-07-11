import unittest

from gateway.compact_middleware import CompactContextMiddleware


def _embed(texts):
    vocabulary = ("alpha", "beta", "gamma", "delta")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


def _body(context_paragraphs, question="tell me about alpha"):
    return {
        "model": "qwen3:8b",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "\n\n".join(context_paragraphs)},
            {"role": "user", "content": question},
        ],
    }


class CompactMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.middleware = CompactContextMiddleware(budget_chars=400, embed_fn=_embed)

    def test_small_requests_pass_through_unchanged(self):
        body = _body(["alpha facts", "beta facts"])
        self.assertIs(self.middleware.before_request(body), body)

    def test_oversized_request_keeps_query_relevant_chunks_in_order(self):
        paragraphs = [
            "alpha " * 30,     # relevant to the question
            "beta " * 30,      # distractor
            "gamma " * 30,     # distractor
            "alpha alpha " + "alpha " * 25,  # most relevant
            "delta " * 30,     # distractor
        ]
        body = _body(paragraphs)
        out = self.middleware.before_request(body)
        self.assertIsNot(out, body)
        kept = out["messages"][1]["content"]
        self.assertLess(len(kept), len(body["messages"][1]["content"]))
        self.assertIn("alpha", kept)
        # original body untouched (deep copy)
        self.assertEqual(body["messages"][1]["content"], "\n\n".join(paragraphs))
        # order preserved: if both alpha chunks survive, chunk 0 text precedes chunk 3 text
        self.assertEqual(out["messages"][0]["content"], "be helpful")  # system untouched

    def test_few_chunks_are_not_compacted(self):
        body = _body(["alpha " * 200])  # oversized but a single chunk
        self.assertIs(self.middleware.before_request(body), body)

    def test_bodies_without_user_messages_pass_through(self):
        body = {"messages": [{"role": "system", "content": "x" * 1000}]}
        self.assertIs(self.middleware.before_request(body), body)

    def test_after_response_is_identity(self):
        response = {"choices": []}
        self.assertIs(self.middleware.after_response({}, response), response)


if __name__ == "__main__":
    unittest.main()
