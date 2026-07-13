import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest import mock

from aioptimizer.sidecar import (
    CREATE_NO_WINDOW,
    SidecarPaths,
    ensure_sidecar,
    launch_sidecar,
)


def _hook_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "plugins"
        / "aioptimizer-codex"
        / "scripts"
        / "user_prompt_submit.py"
    )
    spec = importlib.util.spec_from_file_location("sidecar_hook_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SidecarTests(unittest.TestCase):
    def test_two_concurrent_starters_launch_one_process(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = threading.Event()
            launcher_entered = threading.Event()
            contender_checked = threading.Event()
            allow_launcher = threading.Event()
            launches = []
            results = []

            def health_check():
                is_ready = ready.is_set()
                if threading.current_thread().name == "contender":
                    contender_checked.set()
                return is_ready

            def launcher(paths):
                launches.append(paths)
                launcher_entered.set()
                self.assertTrue(allow_launcher.wait(2))
                return mock.Mock(pid=4321)

            def ensure():
                results.append(ensure_sidecar(
                    directory,
                    startup_timeout_seconds=2,
                    poll_interval_seconds=0.001,
                    health_check=health_check,
                    launcher=launcher,
                    pid_is_alive=lambda pid: True,
                ))

            owner = threading.Thread(target=ensure, name="owner")
            owner.start()
            self.assertTrue(launcher_entered.wait(2))
            contender = threading.Thread(target=ensure, name="contender")
            contender.start()
            self.assertTrue(contender_checked.wait(2))
            ready.set()
            allow_launcher.set()
            owner.join(2)
            contender.join(2)

        self.assertFalse(owner.is_alive())
        self.assertFalse(contender.is_alive())
        self.assertEqual(len(launches), 1)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.ready for result in results))

    def test_health_timeout_fails_open_without_sending_prompt(self):
        module = _hook_module()
        context_request = mock.Mock()
        prompt = "private prompt that must not enter the receipt"
        with tempfile.TemporaryDirectory() as directory:
            def timed_out_sidecar(workspace, **kwargs):
                return ensure_sidecar(
                    workspace,
                    startup_timeout_seconds=0,
                    health_check=lambda: False,
                    launcher=lambda paths: mock.Mock(pid=9876),
                    pid_is_alive=lambda pid: False,
                )

            output, receipt = module.handle_payload(
                {"cwd": directory, "prompt": prompt, "transcript_path": "missing.jsonl"},
                sidecar_ensurer=timed_out_sidecar,
                context_request=context_request,
            )

        self.assertIsNone(output)
        context_request.assert_not_called()
        self.assertEqual(receipt["route"], "sidecar_error")
        self.assertFalse(receipt["injected"])
        self.assertNotIn(prompt, json.dumps(receipt))

    def test_live_pid_file_is_reused_without_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = SidecarPaths.for_workspace(directory)
            paths.runtime_dir.mkdir()
            paths.pid.write_text("2468\n", encoding="ascii")
            health = iter((False, False, True))
            launcher = mock.Mock()
            result = ensure_sidecar(
                directory,
                health_check=lambda: next(health),
                launcher=launcher,
                pid_is_alive=lambda pid: pid == 2468,
                startup_timeout_seconds=1,
            )

        self.assertTrue(result.ready)
        self.assertEqual(result.state, "pid_reused")
        self.assertTrue(result.pid_reused)
        launcher.assert_not_called()

    def test_windows_launch_uses_create_no_window_and_workspace_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = SidecarPaths.for_workspace(directory)
            popen = mock.Mock(return_value=mock.Mock(pid=1234))
            launch_sidecar(
                paths,
                platform_name="nt",
                python_executable="python.exe",
                popen=popen,
            )

            command = popen.call_args.args[0]
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["creationflags"], CREATE_NO_WINDOW)
            self.assertNotIn("start_new_session", kwargs)
            self.assertEqual(Path(kwargs["stdout"].name), paths.log)
            self.assertIn(str(paths.ledger), command)
            self.assertIn("--attention-context", command)
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)

    def test_started_process_pid_and_log_live_under_dot_aioptimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            observed = []
            health = iter((False, False, True))

            def launcher(paths):
                observed.append(paths)
                return mock.Mock(pid=1357)

            result = ensure_sidecar(
                directory,
                health_check=lambda: next(health),
                launcher=launcher,
                pid_is_alive=lambda pid: False,
                startup_timeout_seconds=1,
            )
            paths = observed[0]
            pid_text = paths.pid.read_text(encoding="ascii")

        self.assertTrue(result.ready)
        self.assertEqual(result.state, "started")
        self.assertEqual(pid_text, "1357\n")
        self.assertEqual(paths.runtime_dir.name, ".aioptimizer")


if __name__ == "__main__":
    unittest.main()
