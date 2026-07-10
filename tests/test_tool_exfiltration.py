import unittest

from experiments.brain_runtime.tool_exfiltration import (
    VARIANTS,
    build_documents,
    render_context,
    secret_for,
)


class ToolExfiltrationTests(unittest.TestCase):
    def test_full_context_contains_secret_governed_does_not(self):
        secret = secret_for(101)
        for variant in VARIANTS:
            docs = build_documents(101, variant)
            full_context = render_context(docs, "full")
            governed_context = render_context(docs, "governed")

            self.assertIn(secret, full_context)
            self.assertNotIn(secret, governed_context)

    def test_injection_variant_carries_system_instruction_benign_does_not(self):
        secret = secret_for(101)
        injection_docs = build_documents(101, "injection")
        benign_docs = build_documents(101, "benign")

        injection_private = next(d for d in injection_docs if d["scope"] == "private")
        benign_private = next(d for d in benign_docs if d["scope"] == "private")

        self.assertIn("SYSTEM:", injection_private["text"])
        self.assertIn(secret, injection_private["text"])
        self.assertNotIn("SYSTEM:", benign_private["text"])

    def test_build_documents_has_exactly_one_private_doc(self):
        for variant in VARIANTS:
            docs = build_documents(101, variant)
            self.assertEqual(len(docs), 4)
            private_docs = [d for d in docs if d["scope"] == "private"]
            self.assertEqual(len(private_docs), 1)


if __name__ == "__main__":
    unittest.main()
