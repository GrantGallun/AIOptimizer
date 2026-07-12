import unittest

from experiments.brain_runtime.frontier_context_eval import export_packets, score_packets
from gateway.context_compiler import ConversationCompiler
from tests.test_context_organization_eval import CASE, _embed


class FrontierContextEvalTests(unittest.TestCase):
    def test_export_is_deterministic_provider_neutral_and_complete(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        first = export_packets([CASE], compiler)
        second = export_packets([CASE], compiler)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual({packet["metadata"]["arm"] for packet in first}, {
            "raw", "structured", "attention", "combined",
        })
        self.assertTrue(all(packet["request_id"].startswith("cache-port:") for packet in first))
        self.assertTrue(all(packet["messages"][-1]["content"] for packet in first))

    def test_response_scoring_is_keyed_not_order_dependent(self):
        packets = export_packets([CASE], ConversationCompiler(embed_fn=_embed))
        responses = [
            {"request_id": packet["request_id"], "text": "8800|T0004",
             "prompt_tokens": 10, "completion_tokens": 3, "latency_seconds": .2}
            for packet in reversed(packets)
        ]
        result = score_packets(packets, responses)
        self.assertEqual(len(result["rows"]), 4)
        self.assertTrue(all(row["success"] for row in result["rows"]))
        self.assertTrue(all(summary["end_to_end_success"] == 1.0 for summary in result["summaries"]))

    def test_missing_or_unknown_response_ids_fail_closed(self):
        packets = export_packets([CASE], ConversationCompiler(embed_fn=_embed))
        with self.assertRaisesRegex(ValueError, "missing response"):
            score_packets(packets, [])
        responses = [
            {"request_id": packet["request_id"], "text": "8800|T0004"}
            for packet in packets
        ] + [{"request_id": "unknown", "text": "x"}]
        with self.assertRaisesRegex(ValueError, "unknown response"):
            score_packets(packets, responses)

    def test_export_includes_llm_arm_only_when_precomputed_rewriter_is_supplied(self):
        packets = export_packets(
            [CASE], ConversationCompiler(embed_fn=_embed),
            rewrite_fn=lambda source, query, budget: source[:budget],
        )
        self.assertEqual(len(packets), 5)
        llm = next(packet for packet in packets if packet["metadata"]["arm"] == "llm_rewrite")
        self.assertEqual(llm["metadata"]["rewrite_status"], "ok")


if __name__ == "__main__":
    unittest.main()
