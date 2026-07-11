import io
import json
from pathlib import Path
import tempfile
import unittest

from agent_bus import read_codex


class ReadCodexTests(unittest.TestCase):
    def test_print_messages_handles_non_cp1252_text(self) -> None:
        stream = io.StringIO()

        read_codex.print_messages(
            [("CODEX", "retention ↓ without a crash")],
            tail=20,
            full=True,
            stream=stream,
        )

        self.assertEqual(
            stream.getvalue(),
            "--- CODEX ---\nretention ↓ without a crash\n\n",
        )

    def test_extract_replaces_invalid_utf8(self) -> None:
        payload = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"text": "invalid byte: PLACEHOLDER"}],
            },
        }
        encoded = json.dumps(payload).encode("utf-8").replace(b"PLACEHOLDER", b"\x80")
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Path(tmpdir) / "rollout.jsonl"
            session.write_bytes(encoded + b"\n")

            self.assertEqual(read_codex.extract(session), [("CODEX", "invalid byte: �")])


if __name__ == "__main__":
    unittest.main()
