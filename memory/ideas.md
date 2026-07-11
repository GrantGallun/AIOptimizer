# Ideas

Living queue for Brain Workspace ideas that have not been tested yet.

Update rules:
- Add only actionable or recurring ideas.
- Keep untested hypotheses here until evidence is gathered.
- When an idea is tested, add the result to `hypothesis-graveyard.md` and update or remove the active idea entry.
- Avoid churn; do not log obvious implementation minutiae.

Created: 2026-07-09

---

## ★ North-Star (set 2026-07-11) — Characterize the "determinism dividend"

**Goal:** map WHERE the deterministic kernel genuinely beats best-effort prompting, and where it
does not. HYP-27 showed prompting fully matches the kernel on qwen3:8b (dividend ≈ 0); HYP-28 showed
an idiomatic LangGraph agent is only mildly behind. So the kernel's honest value proposition is not
"determinism beats nothing" — it is a **guarantee + robustness** that should pay off precisely where
probabilistic prompting is unreliable. The north-star is to find and quantify that regime.

**Central metric:** the *determinism dividend* = `full_kernel` recurrence − best-`prompted` recurrence,
measured across two axes:
1. **Model capability** (weak → strong): llama3.2:3b → qwen3:8b → qwen3:14b. Prediction: dividend
   SHRINKS as capability grows (already ≈0 at qwen3:8b; should be LARGE at llama3.2:3b, which cannot
   follow multi-step instructions — HYP-24).
2. **Task difficulty / scale** (more operators, longer memory, harder rules, distractors). Prediction:
   dividend grows as the task stresses retrieval + structure.

**Success = an honest, publishable curve**: "the kernel's advantage over prompting is X on weak models
and →0 on strong ones; it is a reliability floor, most valuable when you can't trust the model to
self-organize." A NULL result (dividend ≈0 everywhere) would honestly retire the kernel's necessity
claim — pre-committed as equally reportable.

### Prioritized loop backlog (Fable executes top-down; re-rank after each verdict)
**Re-ranked 2026-07-11 after HYP-29/31:** the capability axis is answered — dividend **−0.517** on
llama3.2:3b (Refuted: invariants that correct behavior burn a weak model's budget; prompts that
shape behavior don't) and **0.000** on qwen3:8b. The dividend on accuracy is ≤0 at both measured
ends. HYP-31's k-confound was refuted (gap survives k-matching). What remains:
1. **HYP-v6 (NEXT — the decisive open question)** — multi-hop composition (prereg v6, fresh seeds
   211/223/227): does forced ORDERED structure improve chained reasoning beyond strong prompting,
   or is the kernel guarantees-only even there? Saturation stop-rule on dev.
2. **HYP-30 (demoted)** — qwen3:14b dividend point. Low information now (curve is flat-to-negative);
   run only if v6 shows a structure effect worth placing on the capability axis.
3. **HYP-29 hidden read** — only if the FINDINGS headline needs the llama dividend confirmed beyond
   dev (effect is 17x with diagnosed mechanism; low priority).
4. **LangGraph reason-node ablation** — explains HYP-28's residual 0.12 gap; fold into v6 learnings.

This north-star supersedes the older activation-steering / privacy threads as the active focus; those
remain buried in the graveyard as settled. — Fable (Claude Opus 4.8)

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
- Status update (2026-07-10): naive mean-pooled version REFUTED on SmolLM2-135M (HYP-AM-01, graveyard). Only worth reviving with a stronger vector derivation (last-token / learned probe / per-layer) on a 1-3B model; the mean-pooled-vector-vs-in-context question is answered (in-context wins at 135M).
- Next test (if revived): last-token or learned-probe memory vector on a 1-3B instruct model, same pre-registered crossover gate.
- Links: IDEA-20260709-03, IDEA-20260709-05, IDEA-20260709-06, HYP-20260709-07, `experiments/activation_memory`, `experiments/activation_memory/PREREGISTRATION.md`
- Last touched: 2026-07-09

### IDEA-20260709-08: Counterfactual multi-worker memory benchmark
- Status: Testing
- Source: Current Codex session
- Summary: Evaluate whether governed shared memory improves real agent work under delayed, contradictory, irrelevant, and private-worker information. Score task outcome, provenance, stale-fact rate, privacy leakage, token cost, and coordination overhead against matched no-memory, append-only, and retrieval baselines.
- Next test: 14B replication (Codex, t0011). The privacy advantage is now DEMONSTRATED (HYP-14): on clean dump/token queries, append_only leaks 5/5 while governed 0/5 on qwen3:8b — resolving the prior "structural but unelicited" caveat. Query style is the key factor (value=0 leaks, dump/token=5/5), which also explained the Codex/Fable discrepancy. Open follow-ups: adversarial/injection queries; whether the pink-elephant boundary (secret-in-prompt defeats governance) generalizes.
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

### HYP-AM-01: (TESTED 2026-07-10 -> graveyard, Refuted) The best memory-delivery channel depends on memory type.
- Refuted on SmolLM2-135M: no layer/coefficient produced the crossover (max interaction gap -2.51 vs +0.5 needed); mean-pooled activation injection barely moves the model, and in-context beats it for BOTH types. See graveyard HYP-AM-01. Does not rule out larger models / stronger vector derivations.
- Links: IDEA-20260709-07

### HYP-CROSS-01: (TESTED 2026-07-10 -> graveyard HYP-17) The governed leak-reduction gap appears on a model susceptible to indirect injection.
- Refuted for indirect injection (llama3.2:3b also resists tool-output injection, 0/5) — but the run confirmed the DIRECT-extraction gap replicates cross-model (append_only leaks the real secret 4-5/5, governed 0/5 on both qwen3:8b and llama3.2:3b). The governed win is now cross-model and scoped to direct extraction. See graveyard HYP-20260710-17.
- Registered: 2026-07-10. qwen3:8b resisted indirect injection at all strengths (HYP-16), so the gap (full leaks, governed 0) could not be shown there. Prediction: a smaller/differently-tuned model (starting llama3.2:3b) WILL leak the tool-output secret under the disguised/authority injection in the `full` arm, while `governed` (scope-filter) stays 0 — demonstrating the gap and confirming governed's value is a real reduction on susceptible models, not only a guarantee on robust ones.
- Test: rerun `tool_exfiltration.py` and `leak_elicitation.py` with `--model llama3.2:3b` (no new code). Confirmed if full > 0 leaks on any injection variant while governed == 0.
- Architecture principle at stake: how much you can trust model injection-resistance vs. needing the structural guarantee — the core argument for governed retrieval.
- Links: HYP-20260710-14, HYP-20260710-16
- — Fable (Claude Opus 4.8)

## Open Questions

- What should count as a strong enough "test" before moving an idea into the graveyard?
- Should the shared cache eventually be a local file store, SQLite database, vector index, graph database, or small service?
- How aggressive should forgetting/decay be for low-utility ideas?
