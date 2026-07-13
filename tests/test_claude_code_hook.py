import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest


class _GatewayHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        self.__class__.requests.append(json.loads(self.rfile.read(size)))
        payload = json.dumps({
            "route": "attention",
            "context": "[R0001|constraint|source:T0001]\nCache latency below 20ms.",
            "output_chars": 61,
            "embedding_cache_hits": 2,
            "embedding_cache_misses": 1,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class ClaudeCodeHookTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.script = (
            self.root
            / "plugins"
            / "aioptimizer-claude-code"
            / "scripts"
            / "user_prompt_submit.py"
        )
        _GatewayHandler.requests = []

    @staticmethod
    def _write_transcript(directory, *, malformed=False):
        path = Path(directory) / "transcript.jsonl"
        if malformed:
            path.write_text('{"type":"user"\n', encoding="utf-8")
            return path
        rows = [
            {"type": "system", "message": {"role": "system", "content": "hidden"}},
            {
                "type": "user",
                "message": {"role": "user", "content": "Cache latency must stay below 20ms."},
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Use the exact cache."},
                        {"type": "tool_use", "id": "tool-1", "name": "ignored"},
                    ],
                },
            },
            {"type": "progress", "data": {"kind": "hook_progress"}},
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def _run(self, workspace, transcript, *, endpoint):
        prompt = "What was the private cache latency requirement?"
        payload = {
            "session_id": "session-private-123",
            "transcript_path": str(transcript),
            "cwd": str(workspace),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
        environment = {
            **os.environ,
            "AIOPTIMIZER_CONTEXT_URL": endpoint,
            "AIOPTIMIZER_CODEX_CONTEXT_CHARS": "4321",
            "AIOPTIMIZER_CODEX_TIMEOUT_SECONDS": "1",
        }
        completed = subprocess.run(
            [sys.executable, str(self.script)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            cwd=workspace,
            env=environment,
            timeout=10,
        )
        return completed, payload

    @staticmethod
    def _unused_endpoint():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        return f"http://127.0.0.1:{port}/optimize/context"

    @staticmethod
    def _receipts(workspace):
        path = Path(workspace) / ".aioptimizer" / "codex_hook_ledger.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_happy_path_emits_additional_context_from_stub_gateway(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _GatewayHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                transcript = self._write_transcript(directory)
                endpoint = f"http://127.0.0.1:{server.server_port}/optimize/context"
                completed, _ = self._run(directory, transcript, endpoint=endpoint)
                receipts = self._receipts(directory)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        hook_output = output["hookSpecificOutput"]
        self.assertEqual(hook_output["hookEventName"], "UserPromptSubmit")
        self.assertIn("Cache latency below 20ms", hook_output["additionalContext"])
        request = _GatewayHandler.requests[-1]
        self.assertEqual(request["output_budget_chars"], 4321)
        self.assertEqual(request["messages"], [
            {"role": "user", "content": "Cache latency must stay below 20ms."},
            {"role": "assistant", "content": "Use the exact cache."},
        ])
        self.assertTrue(receipts[-1]["injected"])

    def test_missing_gateway_fails_open_with_empty_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = self._write_transcript(directory)
            completed, _ = self._run(
                directory, transcript, endpoint=self._unused_endpoint()
            )
            receipt = self._receipts(directory)[-1]

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(receipt["route"], "error")
        self.assertFalse(receipt["injected"])

    def test_malformed_transcript_fails_open(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = self._write_transcript(directory, malformed=True)
            completed, _ = self._run(
                directory, transcript, endpoint=self._unused_endpoint()
            )
            receipt = self._receipts(directory)[-1]

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(receipt["error_type"], "ValueError")
        self.assertFalse(receipt["injected"])

    def test_receipt_contains_ids_and_counts_but_no_content(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = self._write_transcript(directory)
            completed, payload = self._run(
                directory, transcript, endpoint=self._unused_endpoint()
            )
            receipt = self._receipts(directory)[-1]

        self.assertEqual(completed.returncode, 0, completed.stderr)
        serialized = json.dumps(receipt)
        self.assertIn("query_sha256_16", receipt)
        self.assertEqual(receipt["messages"], 2)
        self.assertEqual(receipt["history_chars"], 57)
        self.assertNotIn(payload["prompt"], serialized)
        self.assertNotIn("Cache latency", serialized)
        self.assertNotIn(payload["session_id"], serialized)


if __name__ == "__main__":
    unittest.main()
