# AIOptimizer for Claude Code

This adapter reads Claude Code's visible transcript JSONL during
`UserPromptSubmit`, sends user/assistant text to the local AIOptimizer context
endpoint, and prints additive context only when the compiler selects the
`attention` route. Claude Code authentication remains inside Claude Code.

Install the directory as a Claude Code plugin, or merge the contents of
`settings.example.json` into `~/.claude/settings.json` after replacing the
example checkout path. The equivalent hook snippet is:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python \"C:\\path\\to\\AIOptimizer\\plugins\\aioptimizer-claude-code\\scripts\\user_prompt_submit.py\"",
            "timeout": 45
          }
        ]
      }
    ]
  }
}
```

The hook uses `AIOPTIMIZER_CONTEXT_URL`,
`AIOPTIMIZER_CODEX_CONTEXT_CHARS`, and
`AIOPTIMIZER_CODEX_TIMEOUT_SECONDS`, matching the Codex adapter. If the local
endpoint, transcript, or request is unavailable, the hook exits successfully
without stdout so Claude Code proceeds unchanged. Content-free receipts are
written under the active workspace's `.aioptimizer/` directory.

From the AIOptimizer checkout, run
`python -m aioptimizer.episodes inspect --workspace <active-project>` to check
aggregate cross-layer and eventual-outcome coverage. The inspector never prints
prompt text or opaque episode identifiers.
