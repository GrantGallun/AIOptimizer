# Pre-Registration v16: Does the reorganization advantage GROW with context volume?

Registered: 2026-07-14 by Fable (Claude Fable 5)
Status: Registered (no v16 run yet). LOCAL Ollama only — $0, no subscription usage.
Origin: user's question ("maybe we should consider adding a lot of context"). We have isolated
reorganized-vs-raw context (HYP-33/34/36); we have NOT swept context VOLUME. HYP-34 showed the
advantage is pressure-dependent; this measures whether it scales with sheer volume.

## Design (extends the frozen HYP-33/context_organization_eval harness)

Same task shape (one target fact + attribution distractors + one forbidden value, buried early),
same scorer, same two arms (raw chronology vs attention-organized), matched budget. ONLY the
context VOLUME varies: N_MESSAGES in {40, 80, 160, 320}. Budget held at a FIXED absolute char
ceiling across volumes (so bigger volume = more must be dropped = more pressure). qwen3:8b, local.
The target fact is always planted in the first ~15% of turns (old/buried), never restated.

## Pre-committed prediction + gate (hidden seeds v16 = 1201/1213/1217, read once)

- **H-v16a (scaling)**: the raw-arm success DECREASES monotonically with N_MESSAGES (chronology
  truncation drops the buried fact more as volume grows), while the attention arm stays >= 0.9.
  The gap (attention - raw) INCREASES with volume; report the gap at each N with Wilson CIs.
- **Null / ceiling (equally reportable)**: if raw already fails at N=40 (gap saturated) or attention
  degrades at high volume, report honestly — either bounds the claim.
- Dev seed 20260711 sanity first (raw must not already be at floor at N=40, or widen budget).

This is the "add a lot of context" test done cheaply and in isolation, BEFORE spending any
subscription tokens on the agent-build A/B (v15.1). If the gap widens with volume here, it predicts
where v15.1 must operate. — Fable (Claude Fable 5)
