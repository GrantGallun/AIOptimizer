import unittest

from experiments.brain_runtime.code_context import (
    abstract,
    build_qa,
    extract_functions,
    functions_with_defaults,
    graded,
    select_topk_ce,
)


class CodeContextTests(unittest.TestCase):
    def test_extract_functions_nonempty_and_shaped(self):
        functions = extract_functions(["agent_bus"])

        self.assertTrue(functions)
        for func in functions:
            for key in ("file", "name", "source", "signature", "defaults"):
                self.assertIn(key, func)
            self.assertIsInstance(func["defaults"], dict)

    def test_functions_with_defaults_nonempty(self):
        functions = extract_functions(["agent_bus"])
        with_defaults = functions_with_defaults(functions)

        self.assertTrue(with_defaults)

        dispatch = next((f for f in with_defaults if f["name"] == "dispatch"), None)
        self.assertIsNotNone(dispatch)
        self.assertIn("lease_seconds", dispatch["defaults"])
        self.assertEqual(dispatch["defaults"]["lease_seconds"], 900.0)

    def test_build_qa_matches_real_defaults(self):
        functions = extract_functions(["agent_bus"])
        qa = build_qa(functions, seed=0, n=5)

        self.assertTrue(qa)
        with_defaults = functions_with_defaults(functions)
        by_name = {f["name"]: f for f in with_defaults}

        for item in qa:
            for key in ("func", "param", "answer", "query"):
                self.assertIn(key, item)
            func = by_name[item["func"]]
            real_default = func["defaults"][item["param"]]
            self.assertEqual(item["answer"], str(real_default))
            self.assertIn(item["param"], item["query"])
            self.assertIn(item["func"], item["query"])

    def test_abstract_hides_body_keeps_signature(self):
        functions = extract_functions(["agent_bus"])
        dispatch = next(f for f in functions if f["name"] == "dispatch")
        # sanity check we picked a multi-line body
        self.assertGreater(dispatch["source"].count("\n"), 1)

        abstracted = abstract(dispatch)

        self.assertIn("def ", abstracted)
        self.assertIn(dispatch["signature"], abstracted)
        self.assertIn("body omitted", abstracted)
        self.assertNotIn("lease_expires_at", abstracted)

    def test_graded_lenient_containment(self):
        self.assertTrue(graded("the value is 900", "900"))
        self.assertFalse(graded("nope", "900"))

    def test_select_topk_ce_ranks_relevant(self):
        target = "def dispatch(self, *, lease_seconds=900.0):\n    return lease_seconds"
        others = [
            "def render_context(target, haystack, arm, k):\n    return arm",
            "def build_qa(functions, seed, n):\n    return functions",
            "def abstract(func):\n    return func['signature']",
            "def graded(response, answer):\n    return answer in response",
            "def extract_functions(paths):\n    return paths",
            "def print_matrix(summary, context_sizes):\n    return summary",
            "def build_prompt(context, query):\n    return context + query",
            "def main():\n    return None",
        ]
        try:
            result = select_topk_ce("lease_seconds default", [target] + others, 3)
        except Exception:
            self.skipTest("cross-encoder unavailable")
            return

        self.assertEqual(len(result), 3)
        self.assertIn(target, result)


if __name__ == "__main__":
    unittest.main()
