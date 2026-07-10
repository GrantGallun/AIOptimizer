import unittest

from experiments.brain_runtime.leak_elicitation import STYLES, build_leak_case
from experiments.brain_runtime.local_worker_eval import prompt_for
from experiments.brain_runtime.presentation_ablation import build_context
from experiments.brain_runtime.presentation_ablation_v21 import prompt_for_v21


class LeakElicitationTests(unittest.TestCase):
    def test_append_only_carries_the_leak_probe_for_every_style(self):
        for style in STYLES:
            with self.subTest(style=style):
                case = build_leak_case(101, style)
                context, _ = build_context(case, "append_only")
                self.assertIn(case.forbidden, context)

    def test_value_forward_scopes_out_the_secret_for_every_style(self):
        for style in STYLES:
            with self.subTest(style=style):
                case = build_leak_case(101, style)
                context, _ = build_context(case, "value_forward")
                self.assertNotIn(case.forbidden, context)

    def test_named_prompt_mode_contains_the_forbidden_value(self):
        case = build_leak_case(101, "value")
        prompt = prompt_for(case, "notes")
        self.assertIn(case.forbidden, prompt)

    def test_clean_prompt_mode_never_names_the_forbidden_value(self):
        case = build_leak_case(101, "value")
        prompt = prompt_for_v21(case, "notes")
        self.assertNotIn(case.forbidden, prompt)

    def test_styles_produce_different_queries(self):
        dump = build_leak_case(101, "dump")
        value = build_leak_case(101, "value")
        self.assertNotEqual(dump.query, value.query)


if __name__ == "__main__":
    unittest.main()
