import json, tempfile, unittest
from pathlib import Path
from experiments.brain_runtime.score_v15 import score_arm, summarize

TASK = {"id": "t", "expected_files": ["m.py"],
        "requirements": [{"id": "c", "must_exclude": ["requests.get("], "is_constraint": True},
                         {"id": "f", "must_include": ["def go"]}],
        "ast_checks": [{"kind": "defines", "name": "go"}]}


class ScoreV15Tests(unittest.TestCase):
    def _write(self, root, name, body):
        (root / "t").mkdir(parents=True, exist_ok=True)
        (root / "t" / name).write_text(body, encoding="utf-8")

    def test_pass_fail_and_constraint_retention(self):
        d = Path(tempfile.mkdtemp())
        self._write(d / "A", "m.py", "def go():\n    return 1\n")
        self._write(d / "B", "m.py", "import requests\ndef go():\n    requests.get('x')\n")
        a, b = score_arm(d / "A", [TASK]), score_arm(d / "B", [TASK])
        self.assertTrue(a["rows"][0]["passed"])
        self.assertFalse(b["rows"][0]["passed"])
        s = summarize(a, b)
        self.assertEqual(s["arms"]["A"]["constraint_retention"], 1.0)
        self.assertEqual(s["arms"]["B"]["constraint_retention"], 0.0)
        self.assertEqual(s["paired"]["A_only"], ["t"])

    def test_missing_file_fails(self):
        d = Path(tempfile.mkdtemp())
        (d / "A").mkdir(parents=True)
        self.assertFalse(score_arm(d / "A", [TASK])["rows"][0]["files_present"])

    def test_fixtures_load_and_are_wellformed(self):
        tasks = json.loads(Path("experiments/brain_runtime/v15_build_tasks.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(tasks), 3)
        for t in tasks:
            self.assertTrue({"id", "prompts", "expected_files", "requirements"} <= set(t))
            self.assertGreaterEqual(len(t["prompts"]), 6)


if __name__ == "__main__":
    unittest.main()
