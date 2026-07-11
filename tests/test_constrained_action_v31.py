import unittest

from experiments.brain_runtime import constrained_action_eval_v31 as v31
from experiments.brain_runtime.coala_ollama import ACTION_SCHEMA_CONDITIONAL


class ConditionalSchemaTests(unittest.TestCase):
    def test_schema_is_a_four_branch_oneof_pinning_kind(self):
        self.assertIn("oneOf", ACTION_SCHEMA_CONDITIONAL)
        branches = ACTION_SCHEMA_CONDITIONAL["oneOf"]
        self.assertEqual(len(branches), 4)
        kinds = set()
        for branch in branches:
            self.assertFalse(branch.get("additionalProperties", True))  # forbids extra fields
            self.assertIn("kind", branch["required"])
            kinds.update(branch["properties"]["kind"]["enum"])
        self.assertEqual(kinds, {"retrieve", "reason", "ground", "learn"})


class RenderOnlyTests(unittest.TestCase):
    def test_render_only_is_nondegenerate_and_malformed_free(self):
        payload = v31.run_ollama(model="mock", seed=20260711, render_only=True)
        self.assertEqual(set(payload["arms"]), set(v31.ARMS))
        for arm in v31.ARMS:
            metrics = payload["arms"][arm]
            self.assertEqual(metrics["malformed_action_rate"], 0.0)
            # The mock reasons before grounding, so completion is measurable (not the v3 degeneracy).
            self.assertGreater(metrics["task_completion_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
