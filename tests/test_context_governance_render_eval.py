import unittest

from experiments.brain_runtime.context_governance_render_eval import render_arms, run
from experiments.local_worker.ollama_client import Generation
from gateway.context_compiler import ConversationCompiler
from tests.test_context_organization_eval import CASE, _embed


class ContextGovernanceRenderEvalTests(unittest.TestCase):
    def test_attention_and_governed_context_are_byte_identical(self):
        contexts, diagnostics = render_arms(CASE, ConversationCompiler(embed_fn=_embed))
        self.assertEqual(contexts["attention"].encode(), contexts["governed_attention"].encode())
        self.assertTrue(diagnostics["byte_identical"])
        self.assertTrue(diagnostics["integrity"]["ok"])

    def test_runner_preserves_paired_records_and_scores_both_arms(self):
        class Client:
            def generate_with_metrics(self, prompt, **kwargs):
                return Generation("8800|T0004", 20, 3, 100_000_000, 0)

        payload = run(
            [CASE], Client(), compiler=ConversationCompiler(embed_fn=_embed)
        )
        self.assertEqual(len(payload["rows"]), 2)
        self.assertTrue(all(row["success"] for row in payload["rows"]))
        self.assertEqual(payload["rows"][0]["record_ids"], payload["rows"][1]["record_ids"])
        self.assertTrue(all(row["byte_identical"] for row in payload["rows"]))


if __name__ == "__main__":
    unittest.main()
