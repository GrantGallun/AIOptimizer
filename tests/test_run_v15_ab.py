import json
import tempfile
import unittest
from pathlib import Path

from experiments.brain_runtime.run_v15_ab import (
    collect_task_artifacts,
    load_tasks,
    next_versioned_result,
    run_paired_tasks,
    validate_tasks,
)


TASKS_PATH = Path("experiments/brain_runtime/v15_build_tasks.json")


class RunV15ABTests(unittest.TestCase):
    def test_all_twelve_tasks_pass_frozen_schema_validation(self):
        tasks = load_tasks(TASKS_PATH)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len({task["id"] for task in tasks}), 12)
        for task in tasks:
            self.assertGreaterEqual(len(task["prompts"]), 6)
            self.assertTrue(any(row.get("is_constraint") for row in task["requirements"]))
            self.assertTrue(task["ast_checks"])

    def test_schema_rejects_unsafe_expected_path(self):
        task = json.loads(TASKS_PATH.read_text(encoding="utf-8"))[0]
        task["expected_files"] = ["../escape.py"]
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            validate_tasks([task], expected_count=1)

    def test_collection_uses_arm_task_relative_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "package" / "worker.py"
            source.parent.mkdir()
            source.write_text("def work():\n    return 1\n", encoding="utf-8")
            task = {"id": "layout_task", "expected_files": ["package/worker.py"]}

            result = collect_task_artifacts(workspace, root / "arm", task)

            target = root / "arm" / "layout_task" / "package" / "worker.py"
            self.assertEqual(target.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))
            self.assertEqual(result, {"collected": ["package/worker.py"], "missing": []})

    def test_paired_runner_stubs_agent_calls_and_collects_both_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair_root = root / "pair"
            control_root = root / "control"
            for arm in ("arm-a-plugin-on", "arm-b-plugin-off"):
                seed = pair_root / arm / "workspace"
                seed.mkdir(parents=True)
                (seed / ".gitignore").write_text(".aioptimizer/\n", encoding="utf-8")
            for arm in ("arm-a", "arm-b"):
                (control_root / arm).mkdir(parents=True)

            task = {
                "id": "stub_task",
                "prompts": [f"prompt {index}" for index in range(6)],
                "expected_files": ["built.py"],
            }
            calls = []

            def stub_agent(**kwargs):
                calls.append(kwargs)
                marker = kwargs["codex_home"].name
                (kwargs["workspace"] / "built.py").write_text(
                    f"def built():\n    return {marker!r}\n", encoding="utf-8"
                )
                return {
                    "turns": len(kwargs["prompts"]),
                    "elapsed_seconds": 0.25,
                    "input_tokens": 10,
                    "cached_input_tokens": 2,
                    "output_tokens": 3,
                }

            arm_a, arm_b, records = run_paired_tasks(
                [task],
                pair_root=pair_root,
                control_root=control_root,
                run_root=root / "run",
                agent_runner=stub_agent,
                windows_sandbox="unelevated",
            )

            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call["prompts"] == tuple(task["prompts"]) for call in calls))
            self.assertTrue(all(call["windows_sandbox"] == "unelevated" for call in calls))
            self.assertNotEqual(calls[0]["workspace"], calls[1]["workspace"])
            self.assertEqual(calls[0]["workspace"].name, "workspace")
            self.assertEqual(calls[1]["workspace"].name, "workspace")
            self.assertTrue((arm_a / "stub_task" / "built.py").is_file())
            self.assertTrue((arm_b / "stub_task" / "built.py").is_file())
            self.assertIsNone(records["A"]["stub_task"]["error"])
            self.assertIsNone(records["B"]["stub_task"]["error"])

    def test_versioned_result_skips_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v15_ab_result_v1.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(next_versioned_result(root).name, "v15_ab_result_v2.json")


if __name__ == "__main__":
    unittest.main()
