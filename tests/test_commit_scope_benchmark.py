import unittest

from agent_bus.commit_scope_benchmark_v1 import run_suite, run_trial


class CommitScopeBenchmarkTests(unittest.TestCase):
    def test_legacy_commits_unrelated_files(self):
        row = run_trial(11, "legacy_add_all")
        self.assertTrue(row["target_committed"])
        self.assertEqual(row["unrelated_committed"], 2)
        self.assertFalse(row["staged_other_preserved"])
        self.assertFalse(row["unstaged_other_preserved"])

    def test_declared_write_set_preserves_unrelated_work(self):
        row = run_trial(11, "declared_write_set")
        self.assertTrue(row["target_committed"])
        self.assertEqual(row["unrelated_committed"], 0)
        self.assertTrue(row["staged_other_preserved"])
        self.assertTrue(row["unstaged_other_preserved"])

    def test_suite_aggregates_both_arms(self):
        summaries = {arm["name"]: arm["summary"] for arm in run_suite([11, 23])["arms"]}
        self.assertEqual(summaries["legacy_add_all"]["unrelated_files_committed"], 4)
        self.assertEqual(summaries["declared_write_set"]["unrelated_files_committed"], 0)


if __name__ == "__main__":
    unittest.main()
