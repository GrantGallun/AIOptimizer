import unittest

from agent_bus.workspace_collision_benchmark_v1 import run_leased_trial, run_suite, run_unleased_trial


class WorkspaceCollisionBenchmarkTests(unittest.TestCase):
    def test_unleased_trial_loses_stale_update(self):
        row = run_unleased_trial(11)
        self.assertEqual(row["content"], "base:B")
        self.assertFalse(row["integrity"])
        self.assertEqual(row["lost_updates"], 1)

    def test_leased_trial_serializes_edit_and_retries_once(self):
        row = run_leased_trial(11)
        self.assertEqual(row["content"], "base:AB")
        self.assertTrue(row["integrity"])
        self.assertEqual(row["deferred"], 1)
        self.assertEqual(row["executor_b_calls"], 1)

    def test_suite_reports_both_arms(self):
        payload = run_suite([11, 23])
        summaries = {arm["name"]: arm["summary"] for arm in payload["arms"]}
        self.assertEqual(summaries["unleased"]["integrity_successes"], 0)
        self.assertEqual(summaries["unleased"]["lost_updates"], 2)
        self.assertEqual(summaries["leased_write_sets"]["integrity_successes"], 2)
        self.assertEqual(summaries["leased_write_sets"]["lost_updates"], 0)


if __name__ == "__main__":
    unittest.main()
