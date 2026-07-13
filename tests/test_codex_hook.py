import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from aioptimizer.codex_hook import extract_visible_messages, process_hook, request_context


def _rollout_row(role, text):
    return json.dumps({
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        },
    })


class CodexHookTests(unittest.TestCase):
    def test_local_context_request_allows_encoder_cold_start(self):
        response = mock.MagicMock()
        response.read.return_value = b'{"route":"raw"}'
        context = mock.MagicMock()
        context.__enter__.return_value = response
        with mock.patch("urllib.request.urlopen", return_value=context) as urlopen:
            result = request_context([], "query", 6000)

        self.assertEqual(result, {"route": "raw"})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 30.0)

    def test_hook_launcher_emits_unicode_as_utf8_under_legacy_windows_codepage(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "plugins" / "aioptimizer-codex" / "scripts" / "user_prompt_submit.py"
        loader = (
            "import importlib.util;"
            f"s=importlib.util.spec_from_file_location('hook_launcher',{str(script)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "print('attention \\u2192 context')"
        )
        env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
        completed = subprocess.run(
            [sys.executable, "-c", loader],
            capture_output=True,
            env=env,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.decode("utf-8").strip(), "attention \u2192 context")

    def _transcript(self, directory):
        path = Path(directory) / "rollout.jsonl"
        path.write_text("\n".join([
            _rollout_row("developer", "hidden instruction"),
            _rollout_row("user", "<environment_context>noise</environment_context>"),
            _rollout_row("user", "Cache latency must stay below 20ms."),
            _rollout_row("assistant", "Use the exact cache."),
        ]) + "\n", encoding="utf-8")
        return path

    def test_extracts_only_visible_non_contextual_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            messages = extract_visible_messages(self._transcript(directory))
        self.assertEqual(messages, [
            {"role": "user", "content": "Cache latency must stay below 20ms."},
            {"role": "assistant", "content": "Use the exact cache."},
        ])

    def test_attention_result_becomes_additional_context_without_receipt_text(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = self._transcript(directory)
            payload = {
                "prompt": "What was the cache latency requirement?",
                "transcript_path": str(transcript),
            }

            def optimizer(messages, query, budget):
                self.assertEqual(len(messages), 2)
                self.assertEqual(query, payload["prompt"])
                self.assertEqual(budget, 6000)
                return {
                    "route": "attention",
                    "context": "[R0001|constraint|source:T0001]\nCache latency below 20ms.",
                    "output_chars": 64,
                    "embedding_cache_hits": 3,
                    "embedding_cache_misses": 1,
                }

            output, receipt = process_hook(payload, optimizer=optimizer)

        context = output["hookSpecificOutput"]["additionalContext"]
        self.assertIn("Cache latency below 20ms", context)
        self.assertTrue(receipt["injected"])
        self.assertEqual(receipt["route"], "attention")
        self.assertNotIn("latency requirement", json.dumps(receipt).lower())

    def test_raw_and_service_error_fail_open(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = self._transcript(directory)
            payload = {"prompt": "What stands out?", "transcript_path": str(transcript)}
            raw_output, raw_receipt = process_hook(
                payload,
                optimizer=lambda messages, query, budget: {
                    "route": "raw", "context": "", "output_chars": 0
                },
            )
            error_output, error_receipt = process_hook(
                payload,
                optimizer=lambda messages, query, budget: (_ for _ in ()).throw(
                    ConnectionError("offline")
                ),
            )

        self.assertIsNone(raw_output)
        self.assertFalse(raw_receipt["injected"])
        self.assertIsNone(error_output)
        self.assertEqual(error_receipt["route"], "error")
        self.assertEqual(error_receipt["error_type"], "ConnectionError")


if __name__ == "__main__":
    unittest.main()
