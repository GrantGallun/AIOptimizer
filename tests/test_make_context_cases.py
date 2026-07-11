import json
import unittest

from experiments.brain_runtime.context_organization_eval import validate_case
from experiments.brain_runtime.make_context_cases import generate_cases


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


if __name__ == "__main__":
    unittest.main()
