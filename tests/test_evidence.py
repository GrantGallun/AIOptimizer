import unittest

from aioptimizer.evidence import EvidenceCard, audit_evidence


def _card(**changes):
    value = {
        "experiment_id": "exp-v1",
        "sample_size": 120,
        "task_families": ["real-code", "real-transcript"],
        "models": ["qwen3:8b", "claude-sonnet"],
        "seeds": ["101", "103", "107"],
        "preregistered": True,
        "hidden_split": True,
        "negative_controls": True,
        "independent_review": True,
        "external_data": True,
        "confidence_interval": True,
        "versioned_artifact": True,
        "judge": "deterministic",
    }
    value.update(changes)
    return EvidenceCard.from_mapping(value)


class EvidenceAuditTests(unittest.TestCase):
    def test_complete_card_meets_both_check_sets(self):
        audit = audit_evidence(_card())
        self.assertTrue(audit["meets_internal_validity_checks"])
        self.assertTrue(audit["meets_external_validity_checks"])
        self.assertEqual(audit["limitations"], [])

    def test_weak_card_exposes_each_research_hole(self):
        audit = audit_evidence(_card(
            sample_size=12,
            task_families=["synthetic"],
            models=["qwen3:8b"],
            seeds=["101"],
            preregistered=False,
            hidden_split=False,
            negative_controls=False,
            independent_review=False,
            external_data=False,
            confidence_interval=False,
            versioned_artifact=False,
            judge="encoder",
        ))
        codes = {item["code"] for item in audit["limitations"]}
        self.assertEqual(codes, {
            "small_sample", "few_seeds", "not_preregistered", "no_hidden_split",
            "no_negative_controls", "no_uncertainty", "unversioned_artifact",
            "single_task_family", "single_model", "author_built_only",
            "no_independent_review", "proxy_judge",
        })
        self.assertFalse(audit["meets_internal_validity_checks"])
        self.assertFalse(audit["meets_external_validity_checks"])

    def test_proxy_judge_can_pass_internal_but_not_external_checks(self):
        audit = audit_evidence(_card(judge="model"))
        self.assertTrue(audit["meets_internal_validity_checks"])
        self.assertFalse(audit["meets_external_validity_checks"])
        self.assertEqual([item["code"] for item in audit["limitations"]], ["proxy_judge"])

    def test_rejects_malformed_metadata(self):
        with self.assertRaisesRegex(ValueError, "sample_size"):
            _card(sample_size=True)
        with self.assertRaisesRegex(ValueError, "judge"):
            _card(judge="vibes")
        with self.assertRaisesRegex(ValueError, "models"):
            _card(models="qwen")


if __name__ == "__main__":
    unittest.main()
