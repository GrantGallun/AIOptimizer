import unittest

from gateway.context_compiler import ConversationCompiler


def _embed(texts):
    vocabulary = ("cache", "latency", "memory", "privacy")
    return [[text.lower().count(word) for word in vocabulary] for text in texts]


class ConversationCompilerTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "Never reveal private memory."},
            {"role": "user", "content": "Measure gateway cache latency."},
            {"role": "assistant", "content": "The cache reduced latency."},
            {"role": "user", "content": "How should memory be organized?"},
        ]

    def test_default_compilation_is_lossless_and_source_grounded(self):
        compiled = ConversationCompiler(embed_fn=_embed).compile(self.messages)

        self.assertEqual([r["text"] for r in compiled.records], [m["content"] for m in self.messages])
        self.assertEqual(compiled.records[0]["source_ids"], ["T0001"])
        self.assertEqual(compiled.turns[0]["content"], self.messages[0]["content"])

    def test_local_rewriter_can_emit_typed_records(self):
        def rewrite(turns):
            return [
                {
                    "kind": "decision",
                    "text": "Measure cache latency.",
                    "source_ids": [turns[1]["id"], turns[2]["id"]],
                    "tags": ["gateway", "cache"],
                }
            ]

        compiled = ConversationCompiler(rewrite_fn=rewrite, embed_fn=_embed).compile(self.messages)

        self.assertEqual(compiled.records[0]["kind"], "decision")
        self.assertEqual(compiled.records[0]["source_ids"], ["T0002", "T0003"])

    def test_rewriter_cannot_cite_nonexistent_source(self):
        compiler = ConversationCompiler(
            rewrite_fn=lambda turns: [{"text": "invented", "source_ids": ["T9999"]}],
            embed_fn=_embed,
        )

        with self.assertRaisesRegex(ValueError, "existing source"):
            compiler.compile(self.messages)

    def test_materializer_returns_relevant_records_and_complete_index(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        compiled = compiler.compile(self.messages)

        context = compiler.materialize(compiled, query="cache latency", max_records=2)

        self.assertEqual(context["schema"], "aioptimizer.compiled-context.v1")
        self.assertEqual(len(context["working_set"]), 2)
        self.assertIn("cache", context["working_set"][0]["text"].lower())
        self.assertEqual(len(context["index"]), len(self.messages))


if __name__ == "__main__":
    unittest.main()
