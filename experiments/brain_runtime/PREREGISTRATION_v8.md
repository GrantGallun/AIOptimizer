# Pre-Registration v8: Harnessed stochasticity — does a sample-vote contract match det-query?

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: Registered (no v8 model run yet)

## Motivation (FINDINGS final thesis, v3)

Where determinism buys accuracy, it buys it at *decision boundaries*. v7 proved the first boundary
(deterministic query construction: prompted 0.796 → 0.974). This tests the second pattern the
user's "unrestrict determinism" question opened: **stochastic exploration inside a deterministic
contract** — N independent reasoning samples at temperature 0.7, aggregated by a deterministic
majority vote. If it works, the kernel gains a second portable mechanism orthogonal to det-query
(and the gateway's disagreement-cascade router inherits its evidence base).

## Arms (multihop task, v6.2 settings, qwen3:8b, within-seed)

- **`prompted`**: baseline (single temp-0 pass).
- **`prompted_vote`**: identical, but the REASON step runs 5 samples at temperature 0.7 and the
  answer is the majority vote of parsed integers (deterministic tie-break). t0025 plumbing.
- **`prompted_det_query`**: v7's mechanism, for within-seed comparison of the two contracts.

## Hypotheses (pre-committed, symmetric)

- **H-v8a (vote works)**: `prompted_vote` − `prompted` ≥ 0.10 and Wilson-significant on hidden.
- **H-v8b (contracts compared)**: report `prompted_vote` vs `prompted_det_query` — which boundary
  carries more, at what token cost (vote is ~5× reason tokens; det-query is free). No gate; honest
  comparison.
- **H-v8c (vote adds nothing, equally reportable)**: vote−prompted not significant → sampling
  diversity doesn't rescue what better retrieval fixes; the pattern is retired for this task.

## Cost accounting (required in the writeup)

Report completion tokens per arm. A vote win that costs 5× tokens is a different product decision
than a free det-query win — the gateway's receipts framing applies to research too.

## Splits

- **Dev**: seed 20260711, `prompted_vote` arm only (prompted 0.765 / det_query 0.980 already
  measured on this seed under identical settings). Sanity: non-degenerate, plumbing works (5
  samples visible in metrics).
- **Hidden**: `stats.FRESH_HIDDEN_SEEDS["v8"] = (401, 409, 419)`, all three arms, read once.

## Dogfooding note (dated)

v8 model calls run THROUGH the AIOptimizer Gateway (`/api/generate` passthrough, ExactCache with
the sampled-request bypass, NO compaction — research prompts must not be altered). The gateway
ledger becomes real receipts data; the eval results remain the research record. If the gateway
perturbs results (any harness anomaly), stop and rerun direct — transport must be invisible.

— Fable (Claude Fable 5), 2026-07-11
