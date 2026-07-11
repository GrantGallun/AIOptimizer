import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from agent_bus.board import Board
from agent_bus.codex_bridge import PROMPT_TEMPLATE, build_codex_command, commit_completed_task, publish_presence, resolve_codex
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
    def test_default_command_targets_one_task_without_git_metadata_access(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            command = build_codex_command(
                "codex.exe",
                root=root,
                extra_args=["--full-auto"],
                task_id="t0042",
            )
            self.assertEqual(command[:2], ["codex.exe", "--full-auto"])
            self.assertIn("never", command)
            self.assertIn("workspace-write", command)
            self.assertEqual(command[-2], "exec")
            self.assertIn("t0042 only", command[-1])
            self.assertIn("Do not run git add/commit", command[-1])
            self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", command)
            self.assertNotIn("--add-dir", command)

    def test_prompt_template_requires_task_id(self):
        self.assertIn("{task_id}", PROMPT_TEMPLATE)


class CodexBridgeCommitTests(unittest.TestCase):
    @staticmethod
    def _git(root, *args):
        import subprocess

        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    def test_completed_task_is_committed_by_declared_write_set(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            self._git(tmp, "init", "-b", "main")
            self._git(tmp, "config", "user.email", "bridge@example.invalid")
            self._git(tmp, "config", "user.name", "bridge")
            target = Path(tmp) / "target.txt"
            target.write_text("base")
            self._git(tmp, "add", "-A")
            self._git(tmp, "commit", "-m", "seed")
            board = Board(root)
            task = board.add(op="impl", title="bridge commit", tier="codex", writes=["target.txt"])
            claimed = board.dispatch(tier="codex", worker="codex")
            target.write_text("changed")
            board.submit(task["id"], worker=claimed["owner"], result="done")

            revision = commit_completed_task(root, task["id"])
            self.assertTrue(revision)
            self.assertEqual(self._git(tmp, "show", "HEAD:target.txt").stdout, "changed")

    def test_incomplete_or_undeclared_task_is_not_committed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "agent_bus"
            root.mkdir()
            board = Board(root)
            incomplete = board.add(op="impl", title="queued", tier="codex", writes=["x.txt"])
            undeclared = board.add(op="run", title="no writes", tier="codex")
            self.assertIsNone(commit_completed_task(root, incomplete["id"]))
            self.assertIsNone(commit_completed_task(root, undeclared["id"]))


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
