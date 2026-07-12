"""The product package must never import research code (caught live: a clean-venv
install crashed because compact_middleware imported agent_bus — invisible in-repo)."""
import pathlib
import re
import unittest

FORBIDDEN = re.compile(r"^\s*(from|import)\s+(agent_bus|experiments)\b", re.MULTILINE)


class PackagingIsolationTests(unittest.TestCase):
    def test_no_research_imports_inside_aioptimizer(self):
        offenders = []
        for path in pathlib.Path("aioptimizer").glob("*.py"):
            if FORBIDDEN.search(path.read_text(encoding="utf-8")):
                offenders.append(path.name)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
