# Ideas

Living queue for Brain Workspace ideas that have not been tested yet.

Update rules:
- Add only actionable or recurring ideas.
- Keep untested hypotheses here until evidence is gathered.
- When an idea is tested, add the result to `hypothesis-graveyard.md` and update or remove the active idea entry.
- Avoid churn; do not log obvious implementation minutiae.

Created: 2026-07-09

## Active Ideas

### IDEA-20260709-01: Codex Brain Workspace plugin
- Status: Active
- Source: User request on 2026-07-09
- Summary: Build a Codex plugin that gives agents a brain-like operating loop: active ideas, tested hypotheses, shared memory, and end-of-task memory updates.
- Next test: Start a new Codex thread and use `$brain-workspace` on the next real project task.
- Links: `plugins/brain-workspace`, HYP-20260709-01
- Last touched: 2026-07-09

### IDEA-20260709-02: Governed shared cache for AI workers
- Status: Active
- Source: User request on 2026-07-09
- Summary: Explore a fast shared memory layer for parallel AI workers with scope, provenance, freshness, contradiction handling, and cheap retrieval.
- Next test: Attach Brain Runtime v0 shared-cache semantics to real worker transcripts or a multi-worker toy task.
- Links: HYP-20260709-08, `experiments/brain_runtime`
- Last touched: 2026-07-09

### IDEA-20260709-03: Activation-based memory retrieval
- Status: Active
- Source: User request on 2026-07-09
- Summary: Replace static note-graph recall with an activation model where relevance, recency, utility, confidence, and graph neighbors influence what comes into context.
- Next test: Compare Brain Runtime v0 activation scoring against embedding retrieval or BM25 on messy project-memory transcripts.
- Links: HYP-20260709-08, `experiments/brain_runtime`
- Last touched: 2026-07-09

### IDEA-20260709-04: Benchmark harness for Brain Workspace
- Status: Active
- Source: Current Codex session
- Summary: Build a repeatable evaluation suite with public benchmarks for memory/task performance plus custom deterministic tests for shared-memory correctness, parallelism, provenance, and forgetting.
- Next test: Expand `brain-runtime-v0-synthetic` with generated task seeds, multi-worker handoff, and leak/provenance metrics.
- Links: IDEA-20260709-01, IDEA-20260709-02, IDEA-20260709-03, HYP-20260709-08, `experiments/brain_runtime`
- Last touched: 2026-07-09

### IDEA-20260709-05: Token and activation steering memory
- Status: Active
- Source: Current Codex session
- Summary: Investigate whether storing token/embedding/activation difference vectors can steer agent behavior, separating raw token-ID manipulation from embedding-space retrieval, logit biasing, and hidden-state steering.
- Next test: Replicate the orthogonalized-vector pass with larger hidden/adversarial seeds, add a no-norm-preservation ablation, and test whether preference-margin gains improve generated answer quality.
- Links: IDEA-20260709-03, IDEA-20260709-04, HYP-20260709-02, HYP-20260709-03, HYP-20260709-04, HYP-20260709-05, HYP-20260709-06, HYP-20260709-07
- Last touched: 2026-07-09

### IDEA-20260709-06: Anti-overfit evaluation protocol
- Status: Active
- Source: Current Codex session
- Summary: Prevent Brain Workspace and activation-steering experiments from optimizing against a fixed visible benchmark by using frozen splits, hidden seeds, adversarial generators, negative controls, transfer tests, and pre-registered hypotheses.
- Next test: Add a pass/fail report markdown generator and include larger seeded hidden/adversarial splits before treating the current pass as robust.
- Links: IDEA-20260709-04, IDEA-20260709-05, HYP-20260709-05, HYP-20260709-06, HYP-20260709-07
- Last touched: 2026-07-09

### IDEA-20260709-07: Memory-as-steering vs in-context memory (activation memory)
- Status: Testing
- Source: Current session (novelty-seeking direction: the seam between the steering and Brain Runtime threads)
- Summary: Deliver the same retrieved memory two ways -- in-context text vs a residual-stream steering vector derived identically from the memory text -- and test the pre-registered crossover: activation wins for procedural memory (how to act), in-context wins for factual memory (a specific fact). This fuses IDEA-03 (activation-based retrieval) with the steering thread instead of leaving them separate.
- Next test: Run the `hf` backend on SmolLM2-135M: tune layer/coefficient on `tasks_dev.jsonl` only, then report the single selected config on hidden/adversarial and record the gate verdict. Then repeat on a 1-3B instruct model before any headline claim.
- Links: IDEA-20260709-03, IDEA-20260709-05, IDEA-20260709-06, HYP-20260709-07, `experiments/activation_memory`, `experiments/activation_memory/PREREGISTRATION.md`
- Last touched: 2026-07-09

### IDEA-20260709-08: Counterfactual multi-worker memory benchmark
- Status: Testing
- Source: Current Codex session
- Summary: Evaluate whether governed shared memory improves real agent work under delayed, contradictory, irrelevant, and private-worker information. Score task outcome, provenance, stale-fact rate, privacy leakage, token cost, and coordination overhead against matched no-memory, append-only, and retrieval baselines.
- Next test: (canonical harness is now v2.2, Codex's 5-case privacy/value split — value_forward 25/25 > append_only 21/25 on qwen3:8b, HYP-13). 14B-Q4 replication before any strong claim; AND a leak-ELICITING privacy variant, because privacy-probe leaks were 0 across all policies on qwen3:8b — the governed privacy advantage is structural but unelicited, so the current benchmark cannot demonstrate a leak-reduction win.
- v2.1 status (2026-07-09): CONFIRMED (caveated). Corrected harness (secret not named in prompt; private-scope de-confounded) → value_forward 20/20, 0 leaks, 0 stale, beating append_only 17/20 on qwen3:8b; gate passed. Contradiction mechanism reconfirmed (C0 1/5 → C1 4/5 → C2 5/5). See HYP-20260709-12. Ran end-to-end through the agent-bus board (Sonnet impl → Fable cross-check → qwen run → in-order verdict retirement).
- v2 status (2026-07-09): single-fact rendering fixed the contradiction bottleneck on qwen3:8b (C0 1/5 -> C1 4/5 -> C2 5/5; C2 total 15/20 > append_only 14/20); pre-registered gate did not pass due to a harness-artifact leak. See HYP-20260709-11.
- Previous next-test (done): Pre-register a v2 retrieval-presentation ablation on a new versioned suite: source labels separated from facts, one resolved fact versus conflicting-note context, then rerun local Qwen3 before frontier replication.
- Links: HYP-20260709-09, HYP-20260709-10, `experiments/brain_runtime/multiworker_benchmark.py`, `experiments/brain_runtime/local_worker_eval.py`
- Fable note (2026-07-09): Diagnosed the HYP-10 refutation by case type against the actual rendered rows. Both governed losses are **presentation** failures downstream of correct retrieval (structural benchmark scores governed 20/20). (1) **contradiction** (governed 1/5 vs append_only 4/5): v1 governed renders two conflicting notes (`limit=2`) best-first; the small model has last-item bias and echoes the worse note listed last. (2) **private-scope** (governed 0/5): the model answered the literal string `public-test` — the *source label* from the `- [source] content` format — not the value `zstd`. CORRECTION to my earlier guess: private-scope is NOT ill-posed; the value is reachable and in-scope. Both are fixed by rendering, not retrieval. Pre-registered the fix in `experiments/brain_runtime/PREREGISTRATION_v2.md` (value-forward + single-fact rendering; gate = governed >= append_only at 0 leaks / 0 stale on held-out seeds). — Fable (Claude Opus 4.8)
- Last touched: 2026-07-09 (Fable note added)

### IDEA-20260709-09: Provider-neutral local worker lane
- Status: Testing
- Source: Current Codex session
- Summary: Use a dependency-free local HTTP adapter for inexpensive repeatable worker runs while preserving the same Brain Runtime interface for frontier hosted-model comparisons.
- Next test: Keep Qwen3 8B as the local regression worker; use it to run the pre-registered v2 retrieval-presentation ablation before spending frontier-model calls.
- Links: HYP-20260709-09, HYP-20260709-10, `experiments/local_worker/ollama_client.py`, `experiments/brain_runtime/local_worker_eval.py`
- Last touched: 2026-07-09

## Candidate Hypotheses

### HYP-AM-01 (pre-registered, untested): The best memory-delivery channel depends on memory type -- activation injection beats in-context for procedural memory, and in-context beats activation for factual memory (a crossover interaction with gap >= 0.5 on hidden).
- Registered: 2026-07-09 in `experiments/activation_memory/PREREGISTRATION.md`
- Status: Registered; toy plumbing passes the gate by construction; no confirmatory HF run yet.
- Links: IDEA-20260709-07

## Open Questions

- What should count as a strong enough "test" before moving an idea into the graveyard?
- Should the shared cache eventually be a local file store, SQLite database, vector index, graph database, or small service?
- How aggressive should forgetting/decay be for low-utility ideas?
