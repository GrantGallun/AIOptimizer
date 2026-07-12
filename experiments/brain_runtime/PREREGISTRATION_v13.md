# Pre-Registration v13: Five-arm deterministic input compiler — the cost/quality gate

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: **read once (hidden 1009/1013/1019)** — H-v13b CONFIRMED (attention 1.000 > llm_rewrite 0.900, CIs disjoint, ~0.01% cost); H-v13c null held (0.400=0.400); H-v13a FAILED (combined 0.867, all 8 misses attribution-formatting). See HYP-36.
Provenance: user delegated the compiler phase to Codex; Codex built the typed IR (8ed834e), the
five-arm harness (c890424), the embedding cache (e098d5d), and ran EXPLORATORY dev only (hidden
untouched — dev: raw .35 / structured .35 / attention 1.00 / combined 1.00 answer, .95 e2e /
llm_rewrite .90 at 26,518 input + 3,681 output tokens + 59.9s vs attention's ~0 tokens + 6.8s).
Fable freezes the gate and runs the hidden read, per the standing contract.

## Arms (harness as built: `--v2-five-arms`; no changes after registration)

`raw` / `structured` (typed IR only) / `attention` / `combined` (typed IR + attention) /
`llm_rewrite` (local-LLM preprocessing). Cases: v9 generator, 20/seed; qwen3:8b.

## Pre-committed claims (hidden seeds `FRESH_HIDDEN_SEEDS["v13"] = (1009, 1013, 1019)`)

1. **H-v13a (governance is free)**: `combined` success ≥ `attention` − 0.05 — the typed
   IR/authority layer costs no accuracy (noninferiority; its value is contracts, not accuracy).
2. **H-v13b (deterministic ≥ generative preprocessing)**: `attention` success ≥ `llm_rewrite` −
   0.05, with token/latency costs reported — deterministic compilation is noninferior to LLM
   rewriting at ~0.01% of the generation cost. THE headline claim if it holds.
3. **H-v13c (typing alone is inert, expected null)**: `structured` ≈ `raw` (no significant gap) —
   pre-committing the null so nobody later claims typing boosts accuracy.
4. Leak = 0 in all arms; attribution reported per arm.

Failures of any clause equally reportable. Wilson intervals via stats.py.

— Fable (Claude Fable 5), 2026-07-11

---

## Frontier addendum (2026-07-12, protocol frozen BEFORE reading any responses — dated)

Subscription-frontier replication (user chose $0 path): the 80 dev packets are answered by two
frontier-class models on existing subscriptions — arm A = Codex/GPT (t0036), arm B = a fresh
Sonnet subagent (deliberately not Fable: the gate-reader cannot be an arm). Both see only the
STRIPPED packet file (metadata/answer keys removed by Fable before either arm was engaged).
Scoring: Codex's deterministic scorer, run by Fable, per (model × arm).

Read intent, pre-committed: this is DEV-tier evidence (dev-derived packets, non-reproducible
subscription sessions — no seeds/logprobs). Signal = per-model ordering of arms: if
attention/combined > raw for BOTH frontier models, HYP-36's flagship generalizes upward
(recorded as dev-stage support; a hidden-packet run may follow). If raw ≈ attention for frontier
models, the claim scopes honestly to local/small answerers. Disagreement between the two vendors
is reported, not averaged away. — Fable (Claude Fable 5)

### Frontier protocol incident + v2 (2026-07-12, dated)

Run #1 (both vendors) DISQUALIFIED before any scoring: the Sonnet arm honestly disclosed it
cross-referenced sibling packets — when a raw/structured context lacked the fact, it answered from
the same group's attention variant, visible in the same file. Any single-session run over all arms
has this leak (even unintentionally, via in-context memory of sibling variants). The flaw was in
Fable's runner protocol, not the packet data or either model's integrity — the arm's self-report
is what caught it. **Protocol v2: per-arm isolation** — four packet files, four separate
sessions/agents per vendor, no access to sibling arms or prior arm answers. Disqualified files
retained with _DISQUALIFIED suffix for the record. — Fable (Claude Fable 5)
