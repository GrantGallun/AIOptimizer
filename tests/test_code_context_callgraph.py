import tempfile
import unittest
from pathlib import Path

from experiments.brain_runtime.code_context import extract_call_edges, extract_functions


class CodeContextCallGraphTests(unittest.TestCase):
    def test_extracts_sorted_direct_and_attribute_call_edges(self):
        source = """\
def f():
    g()
    package.h()
    g()

def g():
    h()

def h():
    return 1

def k():
    k()
    unknown()
"""
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "fixture.py").write_text(source, encoding="utf-8")
            functions = extract_functions([directory])

        self.assertEqual(
            extract_call_edges(functions),
            [("f", "g"), ("f", "h"), ("g", "h")],
        )

    def test_excludes_self_unknown_and_unparseable_calls(self):
        functions = [
            {"name": "f", "source": "def f():\n    f()\n    missing()"},
            {"name": "g", "source": "def g(:"},
        ]

        self.assertEqual(extract_call_edges(functions), [])


if __name__ == "__main__":
    unittest.main()
