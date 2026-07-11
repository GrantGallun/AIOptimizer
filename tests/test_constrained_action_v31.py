import unittest

from experiments.brain_runtime import constrained_action_eval_v31 as v31
from experiments.brain_runtime.coala import DecisionContext
from experiments.brain_runtime.coala_ollama import ACTION_SCHEMA_CONDITIONAL, OllamaCoALAAdapter
from experiments.brain_runtime.multihop_eval import _parse_voted_answer
from experiments.local_worker.ollama_client import Generation


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


class SampleVoteTests(unittest.TestCase):
    def test_reason_samples_join_outputs_and_record_each_generation(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def generate_with_metrics(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                return Generation(f"ANSWER={len(self.calls)}", 1, 1, 1, 1)

        client = FakeClient()
        adapter = OllamaCoALAAdapter(
            client,
            model="mock",
            grounding_actions=["answer"],
            reason_samples=3,
            reason_temperature=0.7,
        )
        context = DecisionContext(
            goal="answer composed problem",
            observation="outer(inner(2, 3), 4)",
            scope="project",
            working={},
        )

        output = adapter.reason("compute", context)

        self.assertEqual(output, "ANSWER=1\n---SAMPLE---\nANSWER=2\n---SAMPLE---\nANSWER=3")
        self.assertEqual(adapter.metrics.calls, 3)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(kwargs["temperature"] == 0.7 for _, kwargs in client.calls))

    def test_majority_vote_and_first_tie_break(self):
        self.assertEqual(
            _parse_voted_answer("ANSWER=3\n---SAMPLE---\nANSWER=5\n---SAMPLE---\nANSWER=5"),
            5,
        )
        self.assertEqual(_parse_voted_answer("ANSWER=7\n---SAMPLE---\nANSWER=8"), 7)


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
