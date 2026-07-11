# Pre-Registration v13: Five-arm deterministic input compiler — the cost/quality gate

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: Registered (no v13 hidden read yet)
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
