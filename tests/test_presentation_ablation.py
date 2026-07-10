import unittest

from experiments.brain_runtime.multiworker_benchmark import Case, build_cases
from experiments.brain_runtime.presentation_ablation import (
    CONDITIONS,
    governed_context,
    render_v1,
    render_value_forward,
)


def _case(seed: int, case_id: str) -> Case:
    return next(case for case in build_cases(seed) if case.id == case_id)


class PresentationAblationTests(unittest.TestCase):
    def test_value_forward_avoids_source_label_privacy_bait(self):
        case = _case(11, "private-scope")

        context, _ = governed_context(case, limit=1, render_fn=render_value_forward)

        self.assertIn(case.expected, context)
        self.assertIn("(source:", context)
        self.assertFalse(context.startswith("- ["))
        self.assertNotIn(case.forbidden, context)

    def test_contradiction_limit1_shows_single_resolved_note(self):
        case = _case(11, "contradiction")

        context, _ = governed_context(case, limit=1, render_fn=render_v1)
        note_lines = [line for line in context.splitlines() if line.startswith("- [")]

        self.assertEqual(len(note_lines), 1)
        self.assertIn(case.expected, context)

    def test_contradiction_limit2_shows_two_notes(self):
        case = _case(11, "contradiction")

        context, _ = governed_context(case, limit=2, render_fn=render_v1)
        note_lines = [line for line in context.splitlines() if line.startswith("- [")]

        self.assertEqual(len(note_lines), 2)

    def test_privacy_invariant_holds_across_seeds_and_conditions(self):
        # Only the governed conditions (C0/C1/C2) carry the safety invariant; the
        # append_only reference arm is the known-unsafe baseline being compared against.
        for seed in (11, 23, 37, 41, 59):
            case = _case(seed, "private-scope")
            for condition, (limit, render_fn) in CONDITIONS.items():
                context, _ = governed_context(case, limit=limit, render_fn=render_fn)
                self.assertNotIn(case.forbidden, context, f"privacy leak in condition={condition} seed={seed}")


if __name__ == "__main__":
    unittest.main()
