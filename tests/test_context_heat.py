import unittest

from aioptimizer.context_compiler import ConversationCompiler


VOCABULARY = ("foundation", "current", "cold", "private")


def _embed(texts):
    return [[text.lower().count(word) for word in VOCABULARY] for text in texts]


class ContextHeatTests(unittest.TestCase):
    def setUp(self):
        self.compiler = ConversationCompiler(embed_fn=_embed)
        self.compiled = self.compiler.compile(
            [
                {"role": "assistant", "content": "foundation"},
                {"role": "assistant", "content": "cold"},
                {"role": "user", "content": "current"},
            ]
        )

    def test_zero_weight_is_identical_to_existing_materialization(self):
        baseline = self.compiler.materialize(self.compiled, query="current", max_records=2)
        heated = self.compiler.materialize(
            self.compiled,
            query="current",
            max_records=2,
            heat={"R0001": 100.0},
            heat_weight=0.0,
        )

        self.assertEqual(heated, baseline)

    def test_accumulated_heat_protects_foundational_record(self):
        heat = self.compiler.record_heat(
            self.compiled.records,
            ["foundation", "foundation", "foundation"],
        )

        context = self.compiler.materialize(
            self.compiled,
            query="current",
            max_records=2,
            heat=heat,
            heat_weight=0.5,
        )

        selected_ids = [record["id"] for record in context["working_set"]]
        self.assertIn("R0001", selected_ids)
        self.assertIn("R0003", selected_ids)
        self.assertNotIn("R0002", selected_ids)

    def test_heat_computation_is_deterministic(self):
        queries = ["foundation current", "foundation"]

        first = self.compiler.record_heat(self.compiled.records, queries)
        second = self.compiler.record_heat(self.compiled.records, queries)

        self.assertEqual(first, second)
        self.assertGreater(first["R0001"], first["R0002"])

    def test_privacy_denied_records_get_no_heat_protection(self):
        compiled = self.compiler.compile(
            [
                {"role": "assistant", "content": "private -- never repeat foundation"},
                {"role": "assistant", "content": "cold"},
                {"role": "user", "content": "current"},
            ]
        )

        heat = self.compiler.record_heat(compiled.records, ["private foundation"])
        context = self.compiler.materialize(
            compiled,
            query="current",
            max_records=3,
            heat={**heat, "R0001": 1000.0},
            heat_weight=0.5,
        )

        self.assertEqual(heat["R0001"], 0.0)
        self.assertNotIn("R0001", [record["id"] for record in context["working_set"]])


if __name__ == "__main__":
    unittest.main()
