import unittest

from agent_bus.command_policy_benchmark_v1 import run_suite, run_trial


class CommandPolicyBenchmarkTests(unittest.TestCase):
    def test_legacy_prefix_allows_injected_suffix(self):
        row = run_trial(11, "legacy_shell_prefix")
        self.assertTrue(row["benign_success"])
        self.assertTrue(row["attack_executed"])
        self.assertFalse(row["attack_blocked"])

    def test_typed_policy_preserves_benign_and_blocks_attack(self):
        row = run_trial(11, "typed_argv_policy")
        self.assertTrue(row["benign_success"])
        self.assertFalse(row["attack_executed"])
        self.assertTrue(row["attack_blocked"])
        self.assertIn("control token", row["policy_result"])

    def test_suite_aggregates_both_arms(self):
        summaries = {arm["name"]: arm["summary"] for arm in run_suite([11, 23])["arms"]}
        self.assertEqual(summaries["legacy_shell_prefix"]["attacks_executed"], 2)
        self.assertEqual(summaries["typed_argv_policy"]["attacks_executed"], 0)
        self.assertEqual(summaries["typed_argv_policy"]["benign_successes"], 2)


if __name__ == "__main__":
    unittest.main()
