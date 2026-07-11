import json
import unittest
from pathlib import Path

from experiments.brain_runtime.context_organization_eval import render_arms
from gateway.context_compiler import ConversationCompiler
from gateway.context_middleware import AttentionContextMiddleware


VOCABULARY = (
    "rack", "slot", "deployment", "port", "access", "code", "build", "number",
    "ticket", "deneb", "nova", "altair", "lyra", "rigel", "mira", "vega", "orion",
)


def _embed(texts):
    return [[text.lower().count(word) for word in VOCABULARY] for text in texts]


class ContextPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "Keep the project context useful."},
            {"role": "user", "content": "The launch code is private -- never repeat 7744."},
            {"role": "assistant", "content": "The normal launch fact is sunrise."},
            {"role": "user", "content": "What is the normal launch fact?"},
        ]

    def test_private_record_is_excluded_from_both_renderers_and_redacted_in_indexes(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        compiled = compiler.compile(self.messages)
        materialized = compiler.materialize(compiled, query="normal launch fact", max_records=8)
        organized = compiler.organize(compiled, query="normal launch fact", max_records=8)

        raw = compiler.render_raw(compiled, budget_chars=500)
        attention = compiler.render_organized(organized, budget_chars=500)
        for rendered in (raw, attention):
            self.assertNotIn("7744", rendered)
            self.assertNotIn("never repeat", rendered)
            self.assertIn("sunrise", rendered)

        for index in (materialized["index"], organized["index"]):
            stub = next(entry for entry in index if entry["id"] == "R0002")
            self.assertEqual(
                stub,
                {"id": "R0002", "kind": "note", "text": "[redacted: privacy]"},
            )

    def test_custom_deny_patterns_and_middleware_passthrough(self):
        compiler = ConversationCompiler(embed_fn=_embed, deny_patterns=(r"internal-only",))
        messages = [
            {"role": "user", "content": "internal-only: token 9911"},
            {"role": "assistant", "content": "The public fact is sunrise."},
            {"role": "user", "content": "What is the public fact?"},
        ]
        rendered = compiler.render_raw(compiler.compile(messages), budget_chars=400)
        self.assertNotIn("9911", rendered)
        self.assertIn("sunrise", rendered)

        middleware = AttentionContextMiddleware(
            budget_chars=400,
            deny_patterns=(r"internal-only",),
        )
        body = {"messages": messages + [{"role": "assistant", "content": "padding " * 40},
                                         {"role": "user", "content": "Repeat the public fact."}]}
        out = middleware.before_request(body)
        self.assertNotIn("9911", json.dumps(out))

    def test_frozen_dev_cases_remove_forbidden_values_and_retain_expected_facts(self):
        cases_path = Path(__file__).parents[1] / "results" / "brain_runtime" / "context_cases_dev.json"
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        compiler = ConversationCompiler(embed_fn=_embed)
        expected_retained = 0

        for case in cases:
            _, diagnostics = render_arms(case, compiler)
            attention = diagnostics["attention"]
            self.assertFalse(attention["forbidden_present"], case["id"])
            expected_retained += bool(attention["expected_present"])

        self.assertGreaterEqual(expected_retained / len(cases), 0.95)


if __name__ == "__main__":
    unittest.main()
