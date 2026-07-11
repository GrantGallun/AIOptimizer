import unittest

from intel.reddit_intel import matches, render_markdown, score_post


class RedditIntelTests(unittest.TestCase):
    def test_matches_is_case_insensitive_and_handles_empty_text(self):
        self.assertTrue(matches("Measured LATENCY on my rig", ["latency"]))
        self.assertFalse(matches("unrelated", ["latency"]))
        self.assertFalse(matches(None, ["latency"]))

    def test_score_post_walks_comments_and_applies_formula(self):
        post = {
            "title": "I benchmarked a quantized model",
            "selftext": "It reached 42 tok/s.",
            "score": 10,
            "comments": [
                {"body": "Can confirm, this works for me", "score": 4},
                {
                    "body": "Interesting, can you share more details?",
                    "score": 3,
                    "replies": [{"body": "same here", "score": 2}],
                },
                {"body": "no signal", "score": 100},
            ],
        }
        scored = score_post(post)
        self.assertEqual(scored["agree_upvotes"], 6)
        self.assertEqual(scored["intrigue_upvotes"], 3)
        self.assertEqual(scored["community_score"], 10 + 3 * 6 + 2 * 3)
        self.assertTrue(scored["empirical"])
        self.assertNotIn("community_score", post)

    def test_score_post_can_be_non_empirical(self):
        scored = score_post({"title": "A general question", "score": 1, "comments": []})
        self.assertFalse(scored["empirical"])

    def test_render_markdown_includes_rank_and_empirical_flag(self):
        scored = score_post(
            {
                "title": "Cache result",
                "selftext": "I measured a speedup",
                "subreddit": "LocalLLaMA",
                "score": 7,
                "comments": [{"body": "can confirm", "score": 2}],
            }
        )
        rendered = render_markdown([scored], {"time": "month", "min_score": 5})
        self.assertIn("## 1. Cache result [E]", rendered)
        self.assertIn("r/LocalLLaMA", rendered)
        self.assertIn("**Agreement:** 2 upvotes", rendered)


if __name__ == "__main__":
    unittest.main()
