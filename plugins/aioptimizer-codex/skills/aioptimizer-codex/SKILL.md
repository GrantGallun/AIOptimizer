---
name: aioptimizer-codex
description: Use AIOptimizer's local deterministic compiler and receipts with subscription-backed Codex while keeping OpenAI authentication inside Codex.
---

# AIOptimizer for Codex

This plugin uses Codex's supported `UserPromptSubmit` hook. It reads only the
visible user/assistant transcript tail supplied by Codex, sends it to the
localhost AIOptimizer compiler, and injects additive context only when the
deterministic relevance gate selects attention mode.

Keep the AIOptimizer source checkout installed, configured with `AIOPTIMIZER_HOME`,
or alongside the active project (for example `C:\Code\AIOptimizer` next to
`C:\Code\TalentTrader`), then rely on automatic context:

1. Keep `.aioptimizer/` ignored in the target repository.
2. Submit a prompt normally. The hook checks localhost and lazily starts one hidden,
   attention-enabled AIOptimizer sidecar when needed.
3. Inspect `.aioptimizer/sidecar.log` or the local `/status` endpoint only when diagnosing.
4. From the AIOptimizer checkout, run
   `python -m aioptimizer.episodes inspect --workspace <active-project>` to verify
   that hook, gateway, usage, verification, retry, and terminal-outcome receipts
   are actually joining. The command reports aggregate coverage only and never
   prints prompt text or opaque episode identifiers.

Vague requests and short histories inject nothing. If the local service is not
available or does not become healthy before the bounded startup deadline, the hook
fails open and Codex proceeds normally. Cross-process locking prevents concurrent
hooks from starting duplicate sidecars. Runtime logs and PID state stay under
`.aioptimizer/`. Content-free hook
receipts are appended to `.aioptimizer/codex_hook_ledger.jsonl` in the active
workspace. Local compiler requests allow 30 seconds for an encoder cold start;
set `AIOPTIMIZER_CODEX_TIMEOUT_SECONDS` to a positive number to override it.

This integration does not inspect server-side instructions, modify OAuth,
replace Codex history, or proxy subscription traffic. It improves the visible
context lane available through supported hooks.
