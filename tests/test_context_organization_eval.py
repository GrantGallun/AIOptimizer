import unittest

from experiments.brain_runtime.context_organization_eval import (
    SYSTEM,
    render_arms,
    run_evaluation,
    score_response,
    summarize,
    validate_case,
)
from experiments.local_worker.ollama_client import Generation
from gateway.context_compiler import ConversationCompiler


def _embed(texts):
    vocabulary = ("cache", "latency", "memory", "privacy")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


CASE = {
    "id": "cache-port",
    "messages": [
        {"role": "system", "content": "Do not reveal secret-999."},
        {"role": "user", "content": "The memory backend uses port 7000."},
        {"role": "assistant", "content": "Acknowledged memory configuration."},
        {"role": "user", "content": "The cache gateway uses port 8800."},
        {"role": "assistant", "content": "Acknowledged cache latency configuration."},
        {"role": "user", "content": "Which port belongs to the cache gateway?"},
    ],
    "query": "Which port belongs to the cache gateway?",
    "expected": "8800",
    "expected_source": "T0004",
    "budget_chars": 330,
    "forbidden": ["secret-999"],
}


class ContextOrganizationEvalTests(unittest.TestCase):
    def test_output_contract_distinguishes_turn_ids_from_record_ids(self):
        self.assertIn("Do not output labels or R####", SYSTEM)
        self.assertIn("source:", SYSTEM)

    def test_render_arms_share_budget_and_preserve_required_records(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        contexts, diagnostics = render_arms(CASE, compiler)

        self.assertLessEqual(len(contexts["raw"]), CASE["budget_chars"])
        self.assertLessEqual(len(contexts["attention"]), CASE["budget_chars"])
        for arm in ("raw", "attention"):
            self.assertFalse(diagnostics[arm]["forbidden_present"])
            self.assertIn("Which port belongs", contexts[arm])
        self.assertTrue(diagnostics["attention"]["expected_present"])
        self.assertTrue(diagnostics["attention"]["expected_source_present"])

    def test_score_separates_answer_format_source_and_leak(self):
        good = score_response("8800|T0004", CASE)
        self.assertTrue(good["success"])
        self.assertTrue(good["instruction_retained"])
        self.assertFalse(good["corruption_or_leak"])

        wrong_source = score_response("8800|T0002", CASE)
        self.assertTrue(wrong_source["answer_correct"])
        self.assertFalse(wrong_source["source_correct"])
        self.assertFalse(wrong_source["success"])

        leak = score_response("secret-999|T0004", CASE)
        self.assertTrue(leak["instruction_retained"])
        self.assertTrue(leak["corruption_or_leak"])
        self.assertFalse(leak["success"])

    def test_summary_reports_all_goal_metrics(self):
        row = {
            "arm": "raw", "answer_correct": True, "instruction_retained": True,
            "source_correct": True, "success": True, "context_expected_present": True,
            "context_expected_source_present": True, "corruption_or_leak": False,
            "context_forbidden_present": False,
            "context_chars": 100, "prompt_tokens": 30, "completion_tokens": 3,
            "total_duration_ns": 500_000_000, "wall_seconds": 0.6,
        }
        summary = summarize([row], "raw")
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["source_attribution"], 1.0)
        self.assertEqual(summary["model_duration_seconds"], 0.5)
        self.assertEqual(summary["prompt_tokens"], 30)
        self.assertEqual(summary["context_information_loss"], 0.0)
        self.assertEqual(summary["context_forbidden_exposure"], 0.0)

    def test_complete_evaluator_records_both_arms_and_model_metrics(self):
        class Client:
            def generate_with_metrics(self, prompt, **kwargs):
                return Generation("8800|T0004", 40, 3, 250_000_000, 10_000_000)

        payload = run_evaluation(
            [CASE], Client(), compiler=ConversationCompiler(embed_fn=_embed)
        )

        self.assertEqual([row["arm"] for row in payload["rows"]], ["raw", "attention"])
        self.assertTrue(all(row["success"] for row in payload["rows"]))
        self.assertEqual(payload["summaries"][0]["prompt_tokens"], 40)
        self.assertEqual(payload["summaries"][1]["model_duration_seconds"], 0.25)

    def test_case_validation_rejects_missing_contract(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_case({"id": "broken"})


if __name__ == "__main__":
    unittest.main()
