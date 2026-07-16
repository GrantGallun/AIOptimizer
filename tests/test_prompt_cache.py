import unittest

from aioptimizer.prompt_cache import PromptCacheTelemetryMiddleware, stable_prefix


class PromptCacheTelemetryTests(unittest.TestCase):
    def test_extracts_only_the_static_leading_prefix(self):
        body = {
            "model": "example",
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "messages": [
                {"role": "system", "content": "stable"},
                {"role": "developer", "content": "also stable"},
                {"role": "user", "content": "dynamic"},
                {"role": "system", "content": "too late to be a prefix"},
            ],
            "temperature": 0.7,
        }
        self.assertEqual(stable_prefix(body), {
            "tools": body["tools"],
            "messages": body["messages"][:2],
        })

    def test_repeated_prefix_is_observed_without_mutating_request(self):
        middleware = PromptCacheTelemetryMiddleware()
        first = {
            "messages": [
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "first"},
            ]
        }
        second = {
            "messages": [
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "second"},
            ]
        }
        self.assertIs(middleware.before_request(first), first)
        first_receipt = middleware.receipt_metadata()
        self.assertFalse(first_receipt["candidate_reuse"])
        self.assertIs(middleware.before_request(second), second)
        second_receipt = middleware.receipt_metadata()
        self.assertTrue(second_receipt["candidate_reuse"])
        self.assertEqual(
            first_receipt["prefix_fingerprint"],
            second_receipt["prefix_fingerprint"],
        )
        self.assertEqual(middleware.status_metadata()["candidate_reuses"], 1)

    def test_dynamic_only_prompt_has_no_reuse_candidate(self):
        middleware = PromptCacheTelemetryMiddleware()
        body = {"messages": [{"role": "user", "content": "only dynamic"}]}
        self.assertIs(middleware.before_request(body), body)
        self.assertEqual(middleware.receipt_metadata(), {
            "observed": False,
            "candidate_reuse": False,
        })

    def test_rejects_invalid_capacity(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PromptCacheTelemetryMiddleware(max_prefixes=0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            PromptCacheTelemetryMiddleware(max_prefixes=1.5)


if __name__ == "__main__":
    unittest.main()
