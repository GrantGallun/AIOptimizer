# Developing through AIOptimizer

This folder is configured to run its own local AIOptimizer gateway against
Ollama. In VS Code, run `Tasks: Run Task` and choose
`AIOptimizer: Start local gateway`. Open a new integrated terminal afterward;
it receives:

```powershell
$env:AIOPT_GATEWAY = "http://127.0.0.1:8800"
```

Repo clients that honor `AIOPT_GATEWAY` will then use AIOptimizer automatically.
For an OpenAI-compatible development client, set its base URL to
`http://127.0.0.1:8800/v1`. Native Ollama clients may use
`http://127.0.0.1:8800` and `/api/chat` or `/api/generate`.

The tracked [`aioptimizer.json`](aioptimizer.json) profile enables adaptive
attention selection and exact caching. Generic compaction is disabled so a
low-relevance/vague request stays on broad chronology instead of being compacted
by a later middleware. Ten percent of optimized requests receive raw-shadow
quality receipts. Runtime rows are local and ignored at
`.aioptimizer/development_ledger.jsonl`.

Check the running configuration with the `AIOptimizer: Check health/status`
task. Summarize actual use with `AIOptimizer: Show development receipts`, or:

```powershell
python -m aioptimizer.report .aioptimizer/development_ledger.jsonl
```

The report covers routes, cache hits, latency, provider-reported tokens, shadow
quality, and explicit requirement contracts. It does not automatically route
the Codex or ChatGPT application itself: those hosted clients must expose a
custom base URL or call the compiler through a plugin/hook. The gateway remains
an adapter; the deterministic compiler is also importable from `aioptimizer`.
