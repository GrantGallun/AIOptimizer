import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.brain_runtime.benchmark import run_benchmark
from experiments.brain_runtime.multiworker_benchmark import run_benchmark as run_multiworker_benchmark
from experiments.brain_runtime.runtime import BrainRuntime


class BrainRuntimeTests(unittest.TestCase):
    def test_v15_ab_setup_enables_auto_review_symmetrically(self):
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        if powershell is None:
            self.skipTest("PowerShell is required for the Windows A/B setup scaffold")

        repository = Path(__file__).resolve().parents[1]
        script = repository / "experiments" / "brain_runtime" / "setup_talenttrader_v15_ab.ps1"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            root = temporary / "v15-ab"
            base_home = temporary / "base-codex"
            control_root = temporary / "control-codex"
            (base_home / "plugins").mkdir(parents=True)
            (base_home / "auth.json").write_text("{}\n", encoding="utf-8")
            (base_home / "config.toml").write_text(
                'model = "test-model"\n\n'
                '[plugins."aioptimizer-codex@personal"]\n'
                "enabled = true\n",
                encoding="utf-8",
            )

            command = [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-Root",
                str(root),
                "-Model",
                "test-model",
                "-AIOptimizerHome",
                str(repository),
                "-BaseCodexHome",
                str(base_home),
                "-CodexControlRoot",
                str(control_root),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run(command, check=True, capture_output=True, text=True)

            config_a = (control_root / "arm-a" / "config.toml").read_text(encoding="utf-8")
            config_b = (control_root / "arm-b" / "config.toml").read_text(encoding="utf-8")
            for setting in (
                'approval_policy = "on-request"',
                'approvals_reviewer = "auto_review"',
                'sandbox_mode = "workspace-write"',
            ):
                self.assertEqual(config_a.count(setting), 1)
                self.assertEqual(config_b.count(setting), 1)

            treatment = re.compile(
                r'(?m)(^\[plugins\."aioptimizer-codex@personal"\]\nenabled\s*=\s*)(?:true|false)$'
            )
            self.assertEqual(treatment.sub(r"\1TREATMENT", config_a), treatment.sub(r"\1TREATMENT", config_b))
            self.assertIn("enabled = true", config_a)
            self.assertIn("enabled = false", config_b)
            seed_a = (root / "arm-a-plugin-on" / "workspace" / ".gitignore").read_bytes()
            seed_b = (root / "arm-b-plugin-off" / "workspace" / ".gitignore").read_bytes()
            self.assertEqual(seed_a, seed_b)

    def test_contradiction_prefers_newer_high_confidence_evidence(self):
        runtime = BrainRuntime(decay_half_life=4.0)
        runtime.remember("parser", "Delimiter is comma.", key="delimiter", value="comma", confidence=0.4)
        runtime.tick(4)
        newer = runtime.remember("parser", "Failing test shows delimiter is pipe.", key="delimiter", value="pipe", confidence=0.95)

        self.assertEqual(len(runtime.contradictions), 1)
        self.assertEqual(runtime.contradictions[0].resolved_to, newer.id)
        self.assertIn(newer.id, next(item for item in runtime.working_memory.values() if item.value == "comma").contradicted_by)

    def test_scoped_shared_cache_filters_private_entries(self):
        runtime = BrainRuntime()
        runtime.publish_cache("shared", "Project-visible fact.", scope="project", source="worker-a")
        runtime.publish_cache("secret", "Worker-private fact.", scope="worker-b", source="worker-b")

        visible = runtime.read_cache("fact", scope="worker-a", limit=5)

        self.assertEqual([item.topic for item in visible], ["shared"])

    def test_consolidation_promotes_used_memory_and_forgets_weak_memory(self):
        runtime = BrainRuntime(working_memory_limit=5)
        useful = runtime.remember("json-parser", "Use structured JSON parsing.", confidence=0.9, utility=0.95)
        runtime.remember("scratch", "Maybe use a temporary note.", confidence=0.1, utility=0.1)
        runtime.retrieve("structured JSON parsing")
        runtime.retrieve("structured JSON parsing")
        runtime.tick(20)

        result = runtime.consolidate()

        self.assertGreaterEqual(result["promoted"], 1)
        self.assertIn(useful.id, runtime.long_term_memory)
        self.assertTrue(all(item.topic != "scratch" for item in runtime.working_memory.values()))

    def test_benchmark_brain_beats_vanilla(self):
        payload = run_benchmark()

        self.assertTrue(payload["summary"]["brain_beats_vanilla"])
        self.assertGreater(payload["summary"]["brain_successes"], payload["summary"]["vanilla_successes"])
        self.assertGreaterEqual(payload["summary"]["contradictions"], 1)
        self.assertGreaterEqual(payload["summary"]["task_hooks"], 1)

    def test_seeded_multiworker_benchmark_governs_handoff_and_privacy(self):
        payload = run_multiworker_benchmark([11, 23])
        summaries = {result["policy"]: result["summary"] for result in payload["results"]}

        self.assertTrue(payload["gate"]["governed_beats_append_only"])
        self.assertTrue(payload["gate"]["zero_governed_privacy_leaks"])
        self.assertTrue(payload["gate"]["governed_provenance_advantage"])
        self.assertEqual(summaries["governed_memory"]["success_rate"], 1.0)
        self.assertGreater(summaries["append_only"]["privacy_leaks"], 0)


if __name__ == "__main__":
    unittest.main()
