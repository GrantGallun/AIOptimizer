import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.brain_runtime.analyze_repetition import main, occurrence_accuracy


ROWS = [
    {"operator": "alpha", "correct": True, "first_appearance": True, "arm": "kernel"},
    {"operator": "beta", "correct": False, "first_appearance": True, "arm": "kernel"},
    {"operator": "alpha", "correct": False, "first_appearance": False, "arm": "kernel"},
    {"operator": "beta", "correct": True, "first_appearance": False, "arm": "kernel"},
    {"operator": "alpha", "correct": True, "first_appearance": False, "arm": "kernel"},
    {"operator": "beta", "correct": True, "first_appearance": False, "arm": "kernel"},
    {"operator": "alpha", "correct": False, "first_appearance": False, "arm": "kernel"},
    {"operator": "beta", "correct": False, "first_appearance": False, "arm": "kernel"},
]


class OccurrenceAccuracyTests(unittest.TestCase):
    def test_groups_accuracy_by_each_operators_occurrence(self):
        self.assertEqual(
            occurrence_accuracy(ROWS),
            {1: 0.5, 2: 0.5, 3: 1.0, 4: 0.0},
        )

    def test_empty_rows(self):
        self.assertEqual(occurrence_accuracy([]), {})


class CliTests(unittest.TestCase):
    def run_cli(self, payload, *args):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            output = io.StringIO()
            with patch.object(sys, "argv", ["analyze_repetition.py", str(result_path), *args]):
                with contextlib.redirect_stdout(output):
                    main()
            return output.getvalue()

    def test_arm_filtering(self):
        rows = [
            {"operator": "alpha", "correct": True, "first_appearance": True, "arm": "a"},
            {"operator": "alpha", "correct": False, "first_appearance": True, "arm": "b"},
            {"operator": "alpha", "correct": False, "first_appearance": False, "arm": "a"},
            {"operator": "alpha", "correct": True, "first_appearance": False, "arm": "b"},
        ]
        self.assertEqual(
            self.run_cli({"rows": rows}, "--arm", "a"),
            "occurrence 1: acc=1.000 (n=1)\noccurrence 2: acc=0.000 (n=1)\n",
        )

    def test_arm_filter_is_ignored_when_rows_lack_arm(self):
        rows = [{key: value for key, value in row.items() if key != "arm"} for row in ROWS[:2]]
        self.assertEqual(
            self.run_cli({"rows": rows}, "--arm", "kernel"),
            "occurrence 1: acc=0.500 (n=2)\n",
        )


if __name__ == "__main__":
    unittest.main()
