import json
import unittest

from experiments.brain_runtime.coala import ActionKind, DecisionContext
from experiments.brain_runtime.coala_ollama import ACTION_SCHEMA, OllamaCoALAAdapter
from experiments.local_worker.ollama_client import Generation, OllamaClient


def generation(text):
    return Generation(text, 1, 1, 1, 1)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_with_metrics(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return generation(self.response)


class ConstrainedActionTests(unittest.TestCase):
    def test_client_threads_format_into_request_body(self):
        client = OllamaClient()
        captured = {}

        def request(_path, body=None):
            captured.update(body or {})
            return {"response": "ready"}

        client._request = request
        schema = {"type": "object"}
        client.generate_with_metrics("test", format=schema)
        self.assertEqual(captured["format"], schema)

    def test_malformed_action_is_counted_and_falls_back_to_retrieve(self):
        client = FakeClient("not json")
        adapter = OllamaCoALAAdapter(client, grounding_actions=["answer"])
        context = DecisionContext("goal", "observation", "project", {})

        action = adapter.policy(context)

        self.assertEqual(action.kind, ActionKind.RETRIEVE)
        self.assertEqual(adapter.metrics.malformed_actions, 1)
        self.assertEqual(adapter.metrics.as_dict()["malformed_actions"], 1)

    def test_constrained_policy_passes_frozen_schema(self):
        client = FakeClient('{"kind":"retrieve","query":"x"}')
        adapter = OllamaCoALAAdapter(client, grounding_actions=["answer"], constrained=True)
        context = DecisionContext("goal", "observation", "project", {})

        adapter.policy(context)

        self.assertIs(client.calls[0][1]["format"], ACTION_SCHEMA)
        self.assertEqual(json.loads(json.dumps(ACTION_SCHEMA)), ACTION_SCHEMA)


if __name__ == "__main__":
    unittest.main()
