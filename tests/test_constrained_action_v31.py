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


class DeterministicQueryOverrideTests(unittest.TestCase):
    def test_retrieve_query_is_rebuilt_from_goal_and_observation(self):
        from experiments.brain_runtime.coala import ActionKind, DecisionContext
        from experiments.brain_runtime.coala_ollama import OllamaCoALAAdapter
        from experiments.local_worker.ollama_client import Generation

        class FakeClient:
            def generate_with_metrics(self, prompt, **kwargs):
                return Generation('{"kind":"retrieve","query":"narrow wrong query","limit":3}', 1, 1, 1, 1)

        adapter = OllamaCoALAAdapter(
            FakeClient(), model="mock", grounding_actions=["answer"],
            deterministic_retrieve_query=True,
        )
        context = DecisionContext(goal="answer zlufi of cfure", observation="zlufi(cfure(2, 6), 7)",
                                  scope="project", working={})
        action = adapter.policy(context)
        self.assertIs(action.kind, ActionKind.RETRIEVE)
        self.assertEqual(action.arguments["query"], "answer zlufi of cfure zlufi(cfure(2, 6), 7)")
        self.assertEqual(action.arguments["limit"], 3)  # model's limit is preserved
