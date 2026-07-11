import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent_bus.codex_bridge import resolve_codex


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


if __name__ == "__main__":
    unittest.main()
