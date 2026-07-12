import unittest

from aioptimizer.middleware import ShortCircuit
from aioptimizer.semantic_cache import SemanticCacheMiddleware, _request_text


def _embed(texts):
    vocabulary = ("alpha", "beta", "gamma", "delta")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SemanticCacheMiddlewareTests(unittest.TestCase):
    def test_extracts_last_user_message_or_ollama_prompt(self):
        body = {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "last"},
            ],
            "prompt": "fallback",
        }
        self.assertEqual(_request_text(body), "last")
        self.assertEqual(_request_text({"prompt": "ollama"}), "ollama")

    def test_near_duplicate_hits_and_response_is_independent(self):
        cache = SemanticCacheMiddleware(threshold=0.95, embed_fn=_embed)
        original = {"messages": [{"role": "user", "content": "alpha beta"}]}
        response = {"choices": [{"message": {"content": "cached"}}]}
        cache.after_response(original, response)
        response["choices"][0]["message"]["content"] = "changed"

        near_duplicate = {
            "messages": [{"role": "user", "content": "please: alpha beta"}]
        }
        hit = cache.before_request(near_duplicate)
        self.assertIsInstance(hit, ShortCircuit)
        self.assertEqual(hit.response["choices"][0]["message"]["content"], "cached")
        hit.response["choices"][0]["message"]["content"] = "mutated"
        self.assertEqual(
            cache.before_request(original).response["choices"][0]["message"]["content"],
            "cached",
        )

    def test_unrelated_request_misses(self):
        cache = SemanticCacheMiddleware(threshold=0.9, embed_fn=_embed)
        original = {"prompt": "alpha alpha"}
        cache.after_response(original, {"response": "cached"})
        unrelated = {"prompt": "delta delta"}
        self.assertIs(cache.before_request(unrelated), unrelated)

    def test_sampled_requests_bypass_lookup_and_storage(self):
        cache = SemanticCacheMiddleware(embed_fn=_embed)
        sampled = {"prompt": "alpha", "options": {"temperature": 0.7}}
        cache.after_response(sampled, {"response": "sample"})
        self.assertIs(cache.before_request(sampled), sampled)

        openai_sampled = {
            "messages": [{"role": "user", "content": "beta"}],
            "temperature": 0.5,
        }
        cache.after_response(openai_sampled, {"response": "sample"})
        self.assertIs(cache.before_request(openai_sampled), openai_sampled)

    def test_evicts_least_recently_used_entry(self):
        cache = SemanticCacheMiddleware(
            threshold=0.99, max_entries=2, embed_fn=_embed
        )
        alpha = {"prompt": "alpha"}
        beta = {"prompt": "beta"}
        gamma = {"prompt": "gamma"}
        cache.after_response(alpha, {"response": "a"})
        cache.after_response(beta, {"response": "b"})
        self.assertIsInstance(cache.before_request(alpha), ShortCircuit)
        cache.after_response(gamma, {"response": "g"})

        self.assertIs(cache.before_request(beta), beta)
        self.assertIsInstance(cache.before_request(alpha), ShortCircuit)
        self.assertIsInstance(cache.before_request(gamma), ShortCircuit)

    def test_ttl_expiry_uses_injected_clock(self):
        clock = _FakeClock()
        cache = SemanticCacheMiddleware(
            ttl_seconds=10, clock=clock, embed_fn=_embed
        )
        body = {"prompt": "alpha"}
        cache.after_response(body, {"response": "cached"})
        clock.now = 10
        self.assertIsInstance(cache.before_request(body), ShortCircuit)
        clock.now = 10.01
        self.assertIs(cache.before_request(body), body)

    def test_body_without_text_passes_through_without_embedding(self):
        def fail_if_called(_texts):
            raise AssertionError("embedder should not be called")

        cache = SemanticCacheMiddleware(embed_fn=fail_if_called)
        body = {"messages": [{"role": "assistant", "content": "hello"}]}
        self.assertIs(cache.before_request(body), body)
        response = {"choices": []}
        self.assertIs(cache.after_response(body, response), response)


if __name__ == "__main__":
    unittest.main()
