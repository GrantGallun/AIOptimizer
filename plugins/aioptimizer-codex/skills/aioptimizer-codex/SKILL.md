---
name: aioptimizer-codex
description: Use AIOptimizer's local deterministic compiler and receipts with subscription-backed Codex while keeping OpenAI authentication inside Codex.
---

# AIOptimizer for Codex

This plugin uses Codex's supported `UserPromptSubmit` hook. It reads only the
visible user/assistant transcript tail supplied by Codex, sends it to the
localhost AIOptimizer compiler, and injects additive context only when the
deterministic relevance gate selects attention mode.

Before relying on automatic context:

1. Start AIOptimizer with adaptive attention enabled (`python -m aioptimizer`).
2. Confirm `http://127.0.0.1:8800/status` lists `AttentionContextMiddleware`.
3. Keep `.aioptimizer/` ignored in the target repository.

Vague requests and short histories inject nothing. If the local service is not
available, the hook fails open and Codex proceeds normally. Content-free hook
receipts are appended to `.aioptimizer/codex_hook_ledger.jsonl` in the active
workspace.

This integration does not inspect server-side instructions, modify OAuth,
replace Codex history, or proxy subscription traffic. It improves the visible
context lane available through supported hooks.
