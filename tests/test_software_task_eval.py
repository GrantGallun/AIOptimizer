import json
import unittest

from experiments.brain_runtime.software_task_eval import (
    evaluate_artifact, make_tasks, validate_source,
)


class SoftwareTaskEvalTests(unittest.TestCase):
    def test_tasks_are_real_executable_contracts_with_middle_context_requirements(self):
        tasks = make_tasks()
        self.assertEqual(len(tasks), 4)
        for task in tasks:
            self.assertEqual(task["allowed_path"], "target.py")
            self.assertIn("unittest", task["tests"])
            self.assertEqual(task["messages"][-1]["content"], task["query"])

    def test_safe_source_and_acceptance_execute_only_in_temp_project(self):
        task = make_tasks()[0]
        content = "def choose_mode(scores, threshold=0.5):\n return 'attention' if scores and max(scores) >= threshold else 'raw'\n"
        result = evaluate_artifact(json.dumps({"path":"target.py","content":content}), task)
        self.assertTrue(result["safe_ast"])
        self.assertTrue(result["tests_passed"], result["detail"])

    def test_unsafe_or_unauthorized_artifacts_are_rejected_before_execution(self):
        task = make_tasks()[0]
        unsafe = evaluate_artifact(json.dumps({"path":"target.py","content":"import os\nos.system('x')"}), task)
        wrong = evaluate_artifact(json.dumps({"path":"tests/test_target.py","content":"x=1"}), task)
        self.assertFalse(unsafe["safe_ast"])
        self.assertFalse(wrong["allowed_path"])
        self.assertFalse(wrong["tests_passed"])

    def test_validator_rejects_dynamic_execution(self):
        safe, detail = validate_source("def f(x):\n return eval(x)\n")
        self.assertFalse(safe)
        self.assertIn("eval", detail)


if __name__ == "__main__":
    unittest.main()
