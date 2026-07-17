import json
import tempfile
import time
import unittest
from pathlib import Path

from experiments.brain_runtime.run_v15_ab import (
    check_treatment_integrity,
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
            self.assertTrue(24 <= len(task["prompts"]) <= 40)
            self.assertFalse(any("reminder" in prompt.casefold() for prompt in task["prompts"]))
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
                if marker == "arm-a":
                    ledger = kwargs["workspace"] / ".aioptimizer" / "codex_hook_ledger.jsonl"
                    ledger.parent.mkdir(parents=True, exist_ok=True)
                    ledger.write_text(
                        json.dumps({
                            "ts": time.time(),
                            "history_chars": 6001,
                            "route": "attention",
                            "injected": True,
                        }) + "\n",
                        encoding="utf-8",
                    )
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
                aioptimizer_home=root,
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
            self.assertEqual(
                records["A"]["stub_task"]["treatment_integrity"],
                {"eligible": 1, "treated": 1, "errors": 0, "rate": 1.0,
                 "delivery_failures": 0, "qualified": True},
            )
            self.assertIn("turn_window_start", records["A"]["stub_task"])
            self.assertIn("turn_window_end", records["A"]["stub_task"])
            self.assertNotIn("treatment_integrity", records["B"]["stub_task"])

    def test_missing_ledger_disqualifies_without_crashing_the_run(self):
        # 2026-07-17: a workspace whose hook never wrote a single ledger row crashed the
        # entire multi-task run (FileNotFoundError propagating out of check_treatment_integrity)
        # -- one task's plugin failure took down all the others. Must degrade to a clear
        # disqualification instead.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair_root = root / "pair"
            control_root = root / "control"
            for arm in ("arm-a-plugin-on", "arm-b-plugin-off"):
                seed = pair_root / arm / "workspace"
                seed.mkdir(parents=True)
            for arm in ("arm-a", "arm-b"):
                (control_root / arm).mkdir(parents=True)

            task = {
                "id": "stub_task",
                "prompts": [f"prompt {index}" for index in range(6)],
                "expected_files": ["built.py"],
            }

            def stub_agent(**kwargs):
                # No ledger is ever written for either arm -- simulates the hook never
                # firing at all (not even a sidecar_error row).
                (kwargs["workspace"] / "built.py").write_text("def built():\n    return 1\n",
                                                               encoding="utf-8")
                return {"turns": len(kwargs["prompts"]), "elapsed_seconds": 0.1,
                        "input_tokens": 5, "cached_input_tokens": 0, "output_tokens": 1}

            arm_a, arm_b, records = run_paired_tasks(
                [task],
                pair_root=pair_root,
                control_root=control_root,
                run_root=root / "run",
                agent_runner=stub_agent,
                aioptimizer_home=root,
            )

            self.assertTrue((arm_a / "stub_task" / "built.py").is_file())
            self.assertTrue((arm_b / "stub_task" / "built.py").is_file())
            self.assertEqual(
                records["A"]["stub_task"]["treatment_integrity"],
                {"eligible": 0, "treated": 0, "errors": 0, "rate": None,
                 "delivery_failures": 0, "qualified": False, "ledger_missing": True},
            )
            self.assertIn("codex_hook_ledger.jsonl never appeared", records["A"]["stub_task"]["error"])

    def _check_integrity(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.jsonl"
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            return check_treatment_integrity(
                ledger,
                window_start=100.0,
                window_end=200.0,
                **kwargs,
            )

    def test_treatment_integrity_all_eligible_rows_treated(self):
        result = self._check_integrity([
            {"ts": 100.0, "history_chars": 6001, "route": "attention", "injected": True},
            {"ts": 200.0, "history_chars": 9000, "route": "attention", "injected": True},
        ])

        self.assertEqual(
            result,
            {"eligible": 2, "treated": 2, "errors": 0, "rate": 1.0,
             "delivery_failures": 0, "qualified": True},
        )

    def test_treatment_integrity_low_rate_from_legitimate_declines_still_qualifies(self):
        # Amendment v15.5: the 2026-07-17 replay found ~11.5% treated is the HEALTHY baseline
        # (most eligible turns are correctly declined by the router's own relevance/tail-
        # coverage judgment) -- a low rate must not disqualify on its own.
        result = self._check_integrity([
            {"ts": 120.0, "history_chars": 7000, "route": "attention", "injected": True},
            {"ts": 130.0, "history_chars": 7000, "route": "raw", "route_reason": "covered_by_recent_tail", "injected": False},
            {"ts": 140.0, "history_chars": 7000, "route": "raw", "route_reason": "low_relevance", "injected": False},
        ])

        self.assertEqual(result["eligible"], 3)
        self.assertEqual(result["treated"], 1)
        self.assertAlmostEqual(result["rate"], 1 / 3)
        self.assertEqual(result["delivery_failures"], 0)
        self.assertTrue(result["qualified"])

    def test_treatment_integrity_attention_without_injection_is_a_delivery_failure(self):
        # The router decided to treat (route=attention) but the injection did not happen --
        # a real pipeline bug, unlike a legitimate raw decline.
        result = self._check_integrity([
            {"ts": 120.0, "history_chars": 7000, "route": "attention", "injected": False},
        ])

        self.assertEqual(result["eligible"], 1)
        self.assertEqual(result["treated"], 0)
        self.assertEqual(result["delivery_failures"], 1)
        self.assertFalse(result["qualified"])

    def test_treatment_integrity_unrecognized_route_disqualifies(self):
        # Matches the real 2026-07-17 pilot #1 failure mode: an orphaned/stale sidecar makes
        # every hook call fail open with route="sidecar_error", not the literal "error".
        result = self._check_integrity([
            {"ts": 120.0, "history_chars": 7000, "route": "attention", "injected": True},
            {"ts": 130.0, "history_chars": 100, "route": "sidecar_error", "injected": False},
        ])

        self.assertEqual(result["rate"], 1.0)
        self.assertEqual(result["errors"], 1)
        self.assertFalse(result["qualified"])

    def test_treatment_integrity_zero_eligible_is_inconclusive_not_disqualified(self):
        result = self._check_integrity([
            {"ts": 120.0, "history_chars": 6000, "route": "below_threshold", "injected": False},
            {"ts": 130.0, "route": "below_threshold", "injected": False},
        ])

        self.assertEqual(result["eligible"], 0)
        self.assertIsNone(result["rate"])
        self.assertEqual(result["delivery_failures"], 0)
        self.assertTrue(result["qualified"])

    def test_treatment_integrity_excludes_rows_outside_inclusive_window(self):
        result = self._check_integrity([
            {"ts": 99.9, "history_chars": 7000, "route": "sidecar_error", "injected": False},
            {"ts": 150.0, "history_chars": 7000, "route": "attention", "injected": True},
            {"ts": 200.1, "history_chars": 7000, "route": "raw", "injected": False},
            {"ts": "150", "history_chars": 7000, "route": "sidecar_error", "injected": False},
        ])

        self.assertEqual(
            result,
            {"eligible": 1, "treated": 1, "errors": 0, "rate": 1.0,
             "delivery_failures": 0, "qualified": True},
        )

    def test_resume_reuses_legacy_collected_artifacts_without_rerunning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair_root = root / "pair"
            control_root = root / "control"
            run_root = root / "run"
            for arm in ("arm-a-plugin-on", "arm-b-plugin-off"):
                (pair_root / arm / "workspace").mkdir(parents=True)
            for arm in ("arm-a", "arm-b"):
                (control_root / arm).mkdir(parents=True)
            for arm in ("A", "B"):
                task_dir = run_root / "artifacts" / arm / "stub_task"
                task_dir.mkdir(parents=True)
                (task_dir / "built.py").write_text(f"ARM = {arm!r}\n", encoding="utf-8")

            task = {
                "id": "stub_task",
                "prompts": [f"prompt {index}" for index in range(6)],
                "expected_files": ["built.py"],
            }

            def fail_if_called(**kwargs):
                self.fail(f"resumed artifact unexpectedly reran: {kwargs}")

            arm_a, arm_b, records = run_paired_tasks(
                [task],
                pair_root=pair_root,
                control_root=control_root,
                run_root=run_root,
                agent_runner=fail_if_called,
                model="gpt-5.6-luna",
                windows_sandbox="unelevated",
                resume=True,
            )

            self.assertTrue((arm_a / "stub_task" / "built.py").is_file())
            self.assertTrue((arm_b / "stub_task" / "built.py").is_file())
            self.assertTrue(records["A"]["stub_task"]["recovered_without_telemetry"])
            self.assertTrue(records["B"]["stub_task"]["recovered_without_telemetry"])
            checkpoint = run_root / "runner_checkpoint.jsonl"
            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 3)

            _, _, resumed_again = run_paired_tasks(
                [task],
                pair_root=pair_root,
                control_root=control_root,
                run_root=run_root,
                agent_runner=fail_if_called,
                model="gpt-5.6-luna",
                windows_sandbox="unelevated",
                resume=True,
            )
            self.assertTrue(resumed_again["A"]["stub_task"]["recovered_without_telemetry"])
            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 3)

    def test_resume_rejects_uncheckpointed_incomplete_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair_root = root / "pair"
            control_root = root / "control"
            run_root = root / "run"
            for arm in ("arm-a-plugin-on", "arm-b-plugin-off"):
                (pair_root / arm / "workspace").mkdir(parents=True)
            for arm in ("arm-a", "arm-b"):
                (control_root / arm).mkdir(parents=True)
            (run_root / "artifacts" / "A" / "stub_task").mkdir(parents=True)
            task = {
                "id": "stub_task",
                "prompts": [f"prompt {index}" for index in range(6)],
                "expected_files": ["built.py"],
            }

            with self.assertRaisesRegex(ValueError, "missing expected artifacts"):
                run_paired_tasks(
                    [task],
                    pair_root=pair_root,
                    control_root=control_root,
                    run_root=run_root,
                    agent_runner=lambda **kwargs: {},
                    resume=True,
                )

    def test_versioned_result_skips_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v15_ab_result_v1.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(next_versioned_result(root).name, "v15_ab_result_v2.json")


if __name__ == "__main__":
    unittest.main()
