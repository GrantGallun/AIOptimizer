import unittest

from experiments.brain_runtime.presentation_ablation import build_context
from experiments.brain_runtime.presentation_ablation_v21 import build_cases_v21, prompt_for_v21


class PresentationAblationV21Tests(unittest.TestCase):
    def test_prompt_never_names_the_forbidden_value(self):
        case = next(case for case in build_cases_v21(11) if case.id == "private-scope")
        prompt = prompt_for_v21(case, "notes")
        self.assertNotIn(case.forbidden, prompt)

    def test_append_only_context_still_carries_the_leak_probe(self):
        case = next(case for case in build_cases_v21(11) if case.id == "private-scope")
        context, _ = build_context(case, "append_only")
        self.assertIn(case.forbidden, context)

    def test_value_forward_context_scopes_out_the_private_note(self):
        case = next(case for case in build_cases_v21(11) if case.id == "private-scope")
        context, _ = build_context(case, "value_forward")
        self.assertNotIn(case.forbidden, context)
        self.assertIn(case.expected, context)

    def test_private_scope_query_is_deconfounded_to_the_handoff_query(self):
        cases = build_cases_v21(11)
        private_scope = next(case for case in cases if case.id == "private-scope")
        handoff = next(case for case in cases if case.id == "handoff")
        self.assertEqual(private_scope.query, handoff.query)


if __name__ == "__main__":
    unittest.main()
