import unittest

from experiments.brain_runtime.presentation_ablation import build_context
from experiments.brain_runtime.presentation_ablation_v22 import build_cases_v22


class SplitCasesTests(unittest.TestCase):
    def _cases(self):
        return {c.id: c for c in build_cases_v22(11)}

    def test_five_cases_with_privacy_and_value_separated(self):
        cases = self._cases()
        self.assertEqual(
            set(cases), {"stale-fact", "handoff", "value-recall", "privacy-probe", "contradiction"}
        )

    def test_value_recall_has_no_secret_and_no_forbidden(self):
        vr = self._cases()["value-recall"]
        self.assertIsNone(vr.forbidden)                                  # no leak signal
        self.assertTrue(all(c.scope != "worker-b" for c in vr.claims))   # secret note dropped

    def test_privacy_probe_keeps_secret_as_out_of_scope_distractor(self):
        pp = self._cases()["privacy-probe"]
        self.assertIsNotNone(pp.forbidden)
        self.assertTrue(any(c.scope == "worker-b" for c in pp.claims))   # secret present but out of scope

    def test_privacy_probe_leak_surface_is_live_but_governed_is_safe(self):
        pp = self._cases()["privacy-probe"]
        secret = pp.forbidden
        # append_only dumps the private note -> the leak probe is real
        self.assertIn(secret, build_context(pp, "append_only")[0])
        # governed value-forward scopes it out and renders the value
        gov = build_context(pp, "value_forward")[0]
        self.assertNotIn(secret, gov)
        self.assertIn(pp.expected, gov)


if __name__ == "__main__":
    unittest.main()
