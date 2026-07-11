import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent_bus.codex_bridge import PROMPT, build_codex_command, publish_presence, resolve_codex
from agent_bus.cache import Cache


class CodexBridgeResolutionTests(unittest.TestCase):
    def test_explicit_path_takes_precedence(self):
        with TemporaryDirectory() as tmp:
            binary = Path(tmp) / "custom-codex.exe"
            binary.write_text("")
            with patch("agent_bus.codex_bridge.shutil.which", return_value=None):
                self.assertEqual(resolve_codex(str(binary), environ={}), str(binary.resolve()))

    def test_persistent_environment_override_is_supported(self):
        with TemporaryDirectory() as tmp:
            binary = Path(tmp) / "env-codex.exe"
            binary.write_text("")
            with patch("agent_bus.codex_bridge.shutil.which", return_value=None):
                self.assertEqual(resolve_codex(environ={"CODEX_BIN": str(binary)}), str(binary.resolve()))

    def test_path_lookup_precedes_windows_install_scan(self):
        with TemporaryDirectory() as tmp:
            binary = Path(tmp) / "codex.exe"
            binary.write_text("")
            with patch("agent_bus.codex_bridge.shutil.which", return_value=str(binary)):
                self.assertEqual(resolve_codex(environ={"LOCALAPPDATA": tmp}), str(binary.resolve()))

    def test_windows_install_scan_selects_newest_version(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "OpenAI" / "Codex" / "bin"
            old = root / "old" / "codex.exe"
            new = root / "new" / "codex.exe"
            old.parent.mkdir(parents=True)
            new.parent.mkdir(parents=True)
            old.write_text("")
            new.write_text("")
            now = time.time()
            os.utime(old, (now - 10, now - 10))
            os.utime(new, (now, now))
            with patch("agent_bus.codex_bridge.shutil.which", return_value=None):
                self.assertEqual(resolve_codex(environ={"LOCALAPPDATA": tmp}), str(new.resolve()))

    def test_missing_binary_returns_none(self):
        with patch("agent_bus.codex_bridge.shutil.which", return_value=None):
            self.assertIsNone(resolve_codex(environ={}))


class CodexBridgeCommandTests(unittest.TestCase):
    def test_default_command_keeps_workspace_sandbox_and_adds_only_git_metadata(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            command = build_codex_command(
                "codex.exe",
                root=root,
                extra_args=["--full-auto"],
            )
            self.assertEqual(command[:2], ["codex.exe", "--full-auto"])
            self.assertIn("never", command)
            self.assertIn("workspace-write", command)
            add_dir = command.index("--add-dir")
            self.assertEqual(Path(command[add_dir + 1]), Path(tmp).resolve() / ".git")
            self.assertEqual(command[-2], "exec")
            self.assertEqual(command[-1], PROMPT)
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)

    def test_git_write_can_be_explicitly_disabled(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            command = build_codex_command(
                "codex.exe",
                root=root,
                extra_args=[],
                allow_git_write=False,
            )
            self.assertNotIn("--add-dir", command)


class CodexBridgePresenceTests(unittest.TestCase):
    def test_presence_is_written_to_shared_cache(self):
        with TemporaryDirectory() as tmp:
            cache = Cache(Path(tmp))
            publish_presence(cache, "bridge online/idle; no ready Codex tasks")
            row = cache.get("status.codex")
            self.assertEqual(row["value"], "bridge online/idle; no ready Codex tasks")
            self.assertEqual(row["writer"], "codex")


if __name__ == "__main__":
    unittest.main()
