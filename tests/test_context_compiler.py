import unittest
import json

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
        self.assertEqual(compiled.records[0]["kind"], "instruction")
        self.assertEqual(compiled.records[0]["authority"], "source")
        self.assertTrue(compiled.records[0]["binding"])

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
        self.assertEqual(compiled.records[0]["authority"], "inferred")
        self.assertFalse(compiled.records[0]["binding"])

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

    def test_attention_organization_clusters_topics_and_pins_instructions(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        compiled = compiler.compile(self.messages)

        context = compiler.organize(
            compiled, query="cache latency", similarity_threshold=0.7, max_records=4
        )

        self.assertEqual(context["schema"], "aioptimizer.attention-context.v1")
        self.assertEqual(
            [record["source_ids"] for record in context["pinned"]],
            [["T0001"], ["T0004"]],
        )
        self.assertEqual(
            [record["source_ids"] for record in context["clusters"][0]["records"]],
            [["T0002"], ["T0003"]],
        )
        self.assertEqual(len(context["index"]), 4)

    def test_attention_budget_cannot_remove_latest_user_or_system(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        context = compiler.organize(
            compiler.compile(self.messages), query="cache", max_records=1
        )

        self.assertEqual(len(context["pinned"]), 2)
        self.assertEqual(context["clusters"], [])

    def test_attention_organization_validates_threshold_and_embedding_count(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        compiled = compiler.compile(self.messages)
        with self.assertRaisesRegex(ValueError, "threshold"):
            compiler.organize(compiled, query="cache", similarity_threshold=2.0)

        broken = ConversationCompiler(embed_fn=lambda texts: [[1.0]])
        with self.assertRaisesRegex(ValueError, "one vector"):
            broken.organize(broken.compile(self.messages), query="cache")

    def test_raw_and_organized_renderers_obey_identical_budget_ceiling(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        compiled = compiler.compile(self.messages)
        organized = compiler.organize(compiled, query="cache latency", max_records=4)
        budget = 300

        raw = compiler.render_raw(compiled, budget_chars=budget)
        attention = compiler.render_organized(organized, budget_chars=budget)

        self.assertLessEqual(len(raw), budget)
        self.assertLessEqual(len(attention), budget)
        self.assertIn("Never reveal private memory.", raw)
        self.assertIn("Never reveal private memory.", attention)
        self.assertIn("How should memory be organized?", raw)
        self.assertIn("How should memory be organized?", attention)
        self.assertIn("cache reduced latency", attention)

    def test_renderer_rejects_budget_that_cannot_hold_pinned_records(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        compiled = compiler.compile(self.messages)

        with self.assertRaisesRegex(ValueError, "too small"):
            compiler.render_raw(compiled, budget_chars=10)

    def test_canonical_snapshot_and_hash_are_reproducible(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        hashes = {compiler.fingerprint(compiler.compile(self.messages)) for _ in range(100)}

        self.assertEqual(len(hashes), 1)
        compiled = compiler.compile(self.messages)
        encoded = compiler.canonical_bytes(compiler.canonical_snapshot(compiled))
        self.assertEqual(encoded, compiler.canonical_bytes(json.loads(encoded)))
        self.assertTrue(next(iter(hashes)).startswith("sha256:"))

    def test_compilation_copies_input_and_integrity_audit_preserves_active_request(self):
        messages = [dict(message) for message in self.messages]
        compiler = ConversationCompiler(embed_fn=_embed)
        compiled = compiler.compile(messages)
        messages[-1]["content"] = "mutated later"

        audit = compiler.audit_integrity(compiled)
        self.assertTrue(audit["ok"])
        self.assertTrue(audit["active_request_preserved"])
        self.assertEqual(compiled.turns[-1]["content"], "How should memory be organized?")

    def test_inferred_record_cannot_be_promoted_to_binding(self):
        compiler = ConversationCompiler(
            rewrite_fn=lambda turns: [{
                "kind": "constraint", "text": "Use enterprise framing.",
                "source_ids": [turns[-1]["id"]], "authority": "inferred", "binding": True,
            }],
            embed_fn=_embed,
        )
        with self.assertRaisesRegex(ValueError, "cannot be binding"):
            compiler.compile(self.messages)

    def test_integrity_audit_detects_falsely_authoritative_rewrite(self):
        compiler = ConversationCompiler(
            rewrite_fn=lambda turns: [{
                "kind": "fact", "text": "A changed claim",
                "source_ids": [turns[-1]["id"]], "authority": "source",
            }],
            embed_fn=_embed,
        )
        audit = compiler.audit_integrity(compiler.compile(self.messages))
        self.assertFalse(audit["ok"])
        self.assertIn("source_text_changed", {item["reason"] for item in audit["failures"]})

    def test_matched_pair_has_identical_records_under_one_budget(self):
        compiler = ConversationCompiler(embed_fn=_embed)
        compiled = compiler.compile(self.messages)
        pinned = compiler.pinned_records(compiled)
        pinned_ids = {record["id"] for record in pinned}
        candidates = [record for record in compiled.records if record["id"] not in pinned_ids]

        flat, structured, selected = compiler.matched_record_pair(
            pinned, candidates, budget_chars=340
        )

        self.assertLessEqual(len(flat), 340)
        self.assertLessEqual(len(structured), 340)
        self.assertIn("## INSTRUCTION", structured)
        self.assertEqual(
            {record["id"] for record in selected},
            {record_id for record_id in ("R0001", "R0002", "R0003", "R0004")},
        )
        for record in selected:
            self.assertIn(record["text"], flat)
            self.assertIn(record["text"], structured)

    def test_incremental_embedding_cache_reuses_exact_context(self):
        calls = []
        def embed(texts):
            calls.append(list(texts))
            return _embed(texts)

        compiler = ConversationCompiler(embed_fn=embed)
        compiled = compiler.compile(self.messages)
        compiler.materialize(compiled, query="cache latency", max_records=4)
        first_call_count = len(calls)
        compiler.materialize(compiled, query="cache latency", max_records=4)
        compiler.organize(compiled, query="cache latency", max_records=4)

        self.assertEqual(len(calls), first_call_count)
        stats = compiler.embedding_cache_stats()
        self.assertEqual(stats["misses"], 5)
        self.assertGreaterEqual(stats["hits"], 10)

    def test_embedding_cache_can_be_disabled_and_validates_capacity(self):
        calls = []
        compiler = ConversationCompiler(
            embed_fn=lambda texts: calls.append(list(texts)) or _embed(texts),
            embed_cache_entries=0,
        )
        compiled = compiler.compile(self.messages)
        compiler.materialize(compiled, query="cache", max_records=4)
        compiler.materialize(compiled, query="cache", max_records=4)
        self.assertEqual(len(calls), 2)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            ConversationCompiler(embed_cache_entries=-1)


if __name__ == "__main__":
    unittest.main()
