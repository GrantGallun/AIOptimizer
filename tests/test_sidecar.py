import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from aioptimizer.sidecar import (
    CREATE_NO_WINDOW,
    SidecarPaths,
    ensure_sidecar,
    launch_sidecar,
    process_is_alive,
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
    @staticmethod
    def _free_port():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return listener.getsockname()[1]

    def test_hook_discovers_sibling_source_checkout(self):
        module = _hook_module()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            workspace = parent / "TalentTrader"
            source_root = parent / "AIOptimizer"
            workspace.mkdir()
            (source_root / "aioptimizer").mkdir(parents=True)
            (source_root / "aioptimizer" / "__init__.py").write_text("", encoding="utf-8")

            discovered = module.discover_aioptimizer_home(workspace, environ={})

        self.assertEqual(discovered, source_root.resolve())

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

    def test_launch_uses_explicit_source_root_outside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "TalentTrader"
            source_root = Path(directory) / "AIOptimizer"
            workspace.mkdir()
            source_root.mkdir()
            paths = SidecarPaths.for_workspace(workspace)
            popen = mock.Mock(return_value=mock.Mock(pid=1234))

            launch_sidecar(paths, source_root=source_root, popen=popen)

        self.assertEqual(Path(popen.call_args.kwargs["cwd"]), source_root.resolve())

    def test_real_hook_bootstraps_from_external_workspace(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "plugins" / "aioptimizer-codex" / "scripts" / "user_prompt_submit.py"
        port = self._free_port()
        completed = None
        rows = []
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            parent = Path(directory)
            workspace = parent / "TalentTrader"
            source_root = parent / "AIOptimizer"
            package = source_root / "aioptimizer"
            workspace.mkdir()
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "sidecar.py").write_text(
                (root / "aioptimizer" / "sidecar.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (package / "__main__.py").write_text(
                "import argparse, json, threading\n"
                "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
                "parser = argparse.ArgumentParser(add_help=False)\n"
                "parser.add_argument('--port', type=int, default=8800)\n"
                "args, _ = parser.parse_known_args()\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                "    def reply(self, body):\n"
                "        payload = json.dumps(body).encode()\n"
                "        self.send_response(200); self.send_header('Content-Type', 'application/json')\n"
                "        self.send_header('Content-Length', str(len(payload))); self.end_headers()\n"
                "        self.wfile.write(payload)\n"
                "    def do_GET(self): self.reply({'status': 'ok'})\n"
                "    def do_POST(self):\n"
                "        self.rfile.read(int(self.headers.get('Content-Length', '0')))\n"
                "        self.reply({'route':'below_threshold','context':'','output_chars':0})\n"
                "        threading.Thread(target=self.server.shutdown, daemon=True).start()\n"
                "    def log_message(self, format, *args): pass\n"
                "server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)\n"
                "timer = threading.Timer(5, server.shutdown); timer.daemon = True; timer.start()\n"
                "server.serve_forever(); server.server_close()\n",
                encoding="utf-8",
            )
            payload = {
                "cwd": str(workspace),
                "prompt": "What did we decide?",
                "transcript_path": str(workspace / "missing.jsonl"),
            }
            environment = {
                **os.environ,
                "PYTHONPATH": "",
                "AIOPTIMIZER_HOME": str(source_root),
                "AIOPTIMIZER_SIDECAR_PORT": str(port),
                "AIOPTIMIZER_HEALTH_URL": f"http://127.0.0.1:{port}/health",
                "AIOPTIMIZER_CONTEXT_URL": f"http://127.0.0.1:{port}/optimize/context",
                "AIOPTIMIZER_SIDECAR_STARTUP_SECONDS": "10",
            }
            completed = subprocess.run(
                [sys.executable, str(script)],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                cwd=workspace,
                env=environment,
                timeout=20,
            )
            pid_path = workspace / ".aioptimizer" / "sidecar.pid"
            pid = int(pid_path.read_text(encoding="ascii"))
            ledger = workspace / ".aioptimizer" / "codex_hook_ledger.jsonl"
            rows = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
            ]
            deadline = time.monotonic() + 6
            log_path = workspace / ".aioptimizer" / "sidecar.log"
            probe_path = workspace / ".aioptimizer" / "sidecar.closed"
            log_released = False
            while time.monotonic() < deadline:
                try:
                    log_path.replace(probe_path)
                    probe_path.replace(log_path)
                    log_released = True
                    break
                except PermissionError:
                    time.sleep(0.05)
            self.assertTrue(log_released, "self-terminating sidecar did not release its log")

        self.assertIsNotNone(completed)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertTrue(pid and pid > 0)
        self.assertEqual(rows[-1]["route"], "below_threshold")
        self.assertTrue(rows[-1]["sidecar_ready"])
        self.assertEqual(rows[-1]["sidecar_state"], "started")

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
