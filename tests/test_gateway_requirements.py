import unittest

from aioptimizer.requirements import evaluate_requirements, extract_requirements


class RequirementContractTests(unittest.TestCase):
    def test_extracts_contract_and_removes_private_envelope(self):
        body = {
            "model": "test",
            "aioptimizer": {"requirements": [{
                "id": "format",
                "must_include": ["DONE"],
                "must_exclude": ["SECRET"],
            }]},
        }
        clean, requirements = extract_requirements(body)
        self.assertEqual(clean, {"model": "test"})
        self.assertNotIn("aioptimizer", clean)
        self.assertIn("aioptimizer", body)
        self.assertEqual(requirements[0].id, "format")

    def test_evaluation_is_deterministic_casefolded_and_text_private(self):
        _, requirements = extract_requirements({
            "aioptimizer": {"requirements": [{
                "id": "contract-1",
                "must_include": ["done"],
                "must_exclude": ["private-value"],
            }]}
        })
        receipt = evaluate_requirements("DONE safely", requirements)
        self.assertTrue(receipt["all_passed"])
        self.assertEqual(receipt["passed"], 1)
        self.assertNotIn("done", str(receipt).lower())
        self.assertNotIn("private-value", str(receipt).lower())

    def test_case_sensitive_and_failed_checks_are_counted(self):
        _, requirements = extract_requirements({
            "aioptimizer": {"requirements": [{
                "id": "exact",
                "must_include": ["DONE"],
                "case_sensitive": True,
            }]}
        })
        receipt = evaluate_requirements("done", requirements)
        self.assertFalse(receipt["all_passed"])
        self.assertEqual(receipt["results"][0]["include_passes"], 0)

    def test_invalid_contracts_fail_closed(self):
        invalid = [
            {"aioptimizer": {}},
            {"aioptimizer": {"requirements": []}},
            {"aioptimizer": {"requirements": [{"id": "empty"}]}},
            {"aioptimizer": {"requirements": [
                {"id": "same", "must_include": ["a"]},
                {"id": "same", "must_include": ["b"]},
            ]}},
        ]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(ValueError):
                extract_requirements(body)


if __name__ == "__main__":
    unittest.main()
