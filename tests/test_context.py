import unittest

from agent_bus.context import ContextCompactor


def _embed(texts):
    """Tiny deterministic stand-in for the encoder used by model-free tests."""
    vocabulary = ("alpha", "beta", "gamma", "delta")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


class ContextCompactorTests(unittest.TestCase):
    def setUp(self):
        self.compactor = ContextCompactor(embed_fn=_embed)
        self.chunks = [
            {"id": "alpha", "text": "alpha rule"},
            {"id": "beta", "text": "beta rule"},
            {"id": "gamma", "text": "gamma rule"},
        ]

    def test_select_returns_encoder_ranked_k_chunks(self):
        selected = self.compactor.select("alpha", self.chunks, 2)

        self.assertEqual([chunk["id"] for chunk in selected], ["alpha", "beta"])
        self.assertEqual(len(selected), 2)

    def test_compact_respects_character_budget(self):
        result = self.compactor.compact("alpha", self.chunks, budget_chars=15)

        self.assertLessEqual(sum(len(chunk["text"]) for chunk in result), 15)
        self.assertEqual(result[0]["id"], "alpha")

    def test_compact_appends_abstracted_remainder(self):
        result = self.compactor.compact(
            "alpha",
            self.chunks,
            budget_chars=25,
            abstract_fn=lambda text: "A",
        )

        self.assertTrue(any(chunk.get("abstract") for chunk in result))
        self.assertIn("A", " ".join(chunk["text"] for chunk in result))
        self.assertLessEqual(sum(len(chunk["text"]) for chunk in result), 25)


if __name__ == "__main__":
    unittest.main()
