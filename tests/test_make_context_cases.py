import hashlib
import json
import unittest

from experiments.brain_runtime.context_organization_eval import validate_case
from experiments.brain_runtime.make_context_cases import generate_cases, generate_volume_cases


class MakeContextCasesTests(unittest.TestCase):
    def setUp(self):
        self.cases = generate_cases(7, n_cases=5)

    def test_deterministic(self):
        again = generate_cases(7, n_cases=5)
        self.assertEqual(json.dumps(self.cases, sort_keys=True), json.dumps(again, sort_keys=True))

    def test_every_case_passes_evaluator_schema(self):
        for case in self.cases:
            validate_case(case)
            self.assertEqual(len(case["messages"]), 40)
            self.assertEqual(case["messages"][0]["role"], "system")
            self.assertEqual(case["messages"][-1]["content"], case["query"])

    def test_planting_invariants(self):
        for case in self.cases:
            contents = [m["content"] for m in case["messages"]]
            hits = [i for i, c in enumerate(contents) if case["expected"] in c]
            self.assertEqual(len(hits), 1)
            self.assertEqual(f"T{hits[0] + 1:04d}", case["expected_source"])
            self.assertTrue(5 <= hits[0] + 1 <= 30)
            forbidden = case["forbidden"][0]
            forb_hits = [c for c in contents if forbidden in c]
            self.assertEqual(len(forb_hits), 1)
            self.assertIn("private", forb_hits[0])
            self.assertLess(case["budget_chars"], sum(len(c) for c in contents) // 2)

    def test_frozen_generator_fingerprint_is_unchanged(self):
        payload = json.dumps(
            generate_cases(7, n_cases=5), sort_keys=True, separators=(",", ":")
        ).encode()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "90601b14e4aeed4220defc536c3db8babc9a331a036e29717e1e5abe0c72c13d",
        )

    def test_volume_cases_are_deterministic(self):
        cases = generate_volume_cases(19, n_cases=5, n_messages=160, budget_chars=1200)
        again = generate_volume_cases(19, n_cases=5, n_messages=160, budget_chars=1200)
        self.assertEqual(json.dumps(cases, sort_keys=True), json.dumps(again, sort_keys=True))

    def test_volume_cases_pass_schema_at_every_registered_volume(self):
        for n_messages in (40, 80, 160, 320):
            with self.subTest(n_messages=n_messages):
                cases = generate_volume_cases(
                    23, n_cases=3, n_messages=n_messages, budget_chars=1400
                )
                for case in cases:
                    validate_case(case)
                    self.assertEqual(len(case["messages"]), n_messages)
                    self.assertEqual(case["messages"][0]["role"], "system")
                    self.assertEqual(case["messages"][-1]["content"], case["query"])
                    self.assertEqual(case["budget_chars"], 1400)

    def test_volume_target_is_planted_once_in_first_fifteen_percent(self):
        for n_messages in (40, 80, 160, 320):
            cases = generate_volume_cases(29, n_cases=8, n_messages=n_messages)
            for case in cases:
                contents = [message["content"] for message in case["messages"]]
                hits = [index + 1 for index, content in enumerate(contents)
                        if case["expected"] in content]
                self.assertEqual(len(hits), 1)
                self.assertGreaterEqual(hits[0], 2)
                self.assertLessEqual(hits[0], int(n_messages * 0.15))
                self.assertEqual(case["expected_source"], f"T{hits[0]:04d}")

    def test_volume_budget_uses_fixed_ceiling_or_fraction_default(self):
        for n_messages in (40, 80, 160, 320):
            fixed = generate_volume_cases(
                31, n_cases=2, n_messages=n_messages, budget_chars=1777
            )
            self.assertTrue(all(case["budget_chars"] == 1777 for case in fixed))

            fractional = generate_volume_cases(31, n_cases=2, n_messages=n_messages)
            for case in fractional:
                full_chars = sum(len(message["content"]) for message in case["messages"])
                self.assertEqual(case["budget_chars"], int(0.45 * full_chars))


if __name__ == "__main__":
    unittest.main()
