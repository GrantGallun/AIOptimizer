import unittest
from pathlib import Path

from experiments.brain_runtime.confaide_governed import (
    DEFAULT_DATA_PATH,
    filter_transcript,
    leaked,
    parse_records,
    secret_keywords,
)


class ConfaideGovernedTests(unittest.TestCase):
    def _records(self):
        text = Path(DEFAULT_DATA_PATH).read_text(encoding="utf-8")
        return parse_records(text)

    def test_parse_records_yields_twenty_records_with_topic_and_transcript(self):
        records = self._records()
        self.assertEqual(len(records), 20)
        for record in records:
            self.assertTrue(record["topic"])
            self.assertTrue(record["transcript"])

    def test_scope_filter_removes_lines_carrying_the_secret_keywords(self):
        records = self._records()
        for record in records:
            keywords = secret_keywords(record["topic"])
            if not keywords:
                continue
            with self.subTest(topic=record["topic"]):
                self.assertTrue(leaked(record["transcript"], keywords))
                filtered = filter_transcript(record["transcript"], keywords)
                self.assertFalse(leaked(filtered, keywords))

    def test_surprise_birthday_record_is_filtered_correctly(self):
        records = self._records()
        target = next(r for r in records if "surprise birthday" in r["topic"])
        keywords = secret_keywords(target["topic"])
        self.assertIn("surprise", [k.lower() for k in keywords])
        self.assertIn("birthday", [k.lower() for k in keywords])
        self.assertTrue(leaked(target["transcript"], keywords))
        filtered = filter_transcript(target["transcript"], keywords)
        self.assertFalse(leaked(filtered, keywords))

    def test_leaked_true_when_keyword_present(self):
        self.assertTrue(leaked("we are planning a surprise party", ["surprise"]))

    def test_leaked_false_when_keyword_absent(self):
        self.assertFalse(leaked("project deadline is friday", ["surprise"]))


if __name__ == "__main__":
    unittest.main()
