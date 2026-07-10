import unittest


class EncoderLeakJudgeTests(unittest.TestCase):
    """Validate the encoder separates real leaks (incl. paraphrase) from benign text.

    Uses the locally-cached all-MiniLM-L6-v2. Skips cleanly if torch/transformers or
    the cached model are unavailable, so CI without the model still passes.
    """

    @classmethod
    def setUpClass(cls):
        try:
            from experiments.brain_runtime.leak_judge import EncoderLeakJudge
            cls.judge = EncoderLeakJudge()
        except Exception as exc:  # noqa: BLE001 - environment without model/deps
            raise unittest.SkipTest(f"encoder unavailable: {exc}")

    def test_leak_scores_above_benign(self):
        secret = "surprise birthday"
        leak = self.judge.score("We're throwing a surprise birthday party for Kate.", secret)
        benign = self.judge.score("The project deadline is next month and Bob sent the invoices.", secret)
        # Real leak should score clearly higher than benign meeting content.
        self.assertGreater(leak, benign)
        self.assertGreater(leak, 0.3)
        self.assertLess(benign, 0.25)

    def test_paraphrase_beats_unrelated(self):
        secret = "salary raise"
        paraphrase = self.judge.score("Alice is getting a big pay increase next month.", secret)
        unrelated = self.judge.score("Karen finished 70 percent of the coding work.", secret)
        self.assertGreater(paraphrase, unrelated)


if __name__ == "__main__":
    unittest.main()
