# Pre-Registration v7: Is the kernel's multi-hop edge just deterministic query construction?

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: **H-v7a CONFIRMED** (hidden 307/311/313): prompted 0.796 / det_query 0.974 / kernel 1.000 — the edge reduces to deterministic query construction (+0.178 significant; within 0.03 of kernel). See HYP-31.

## Motivation

HYP-30 (v6 hidden): full_kernel 0.987 vs best-effort prompted 0.884 — a Wilson-significant +0.103
under composition. The dev diagnosis attributed the prompted arm's misses largely to *self-chosen
retrieval queries that miss one of the two needed rules*, while the kernel's fallback builds the
query deterministically from goal+observation. If that ONE mechanism carries the edge, the kernel's
entire accuracy contribution reduces to a bolt-on line any agent (including LangGraph) can adopt —
the sharpest, most portable claim available. If it doesn't, the invariants/completion guarantee
carry the rest.

## Arms (multihop task, identical settings to v6.2 — 256 reason tokens, budget 4, qwen3:8b)

- **`prompted`**: v6.2 baseline (composition-aware strong prompt, model-chosen queries).
- **`prompted_det_query`**: IDENTICAL, except every model-emitted RETRIEVE has its query rebuilt
  deterministically as `"{goal} {observation}"` (model's limit preserved). No invariants, no
  constrained decoding — pure query-construction isolation.
- **`full_kernel`**: v6 reference, rerun on the same fresh seeds for within-seed comparability.

## Hypotheses (pre-committed, symmetric)

- **H-v7a (mechanism carries the edge)**: on hidden, `prompted_det_query` − `prompted` is
  Wilson-significant AND `prompted_det_query` ≥ `full_kernel` − 0.03. → The kernel's multi-hop
  accuracy edge reduces to deterministic query construction (bolt-on mechanism; headline finding).
- **H-v7b (mechanism is partial)**: det_query improves on prompted significantly but stays
  > 0.03 below full_kernel → query construction is part of the edge; completion/invariants carry
  the rest. Report the split.
- **H-v7c (mechanism is nothing, equally reportable)**: det_query − prompted not significant →
  the dev attribution was wrong; the edge lives in completion/invariants alone.

## Splits

- **Dev**: seed 20260711, `prompted_det_query` arm only (prompted 0.765 / kernel 1.000 already
  measured on this seed under v6.2 settings). Sanity: det_query should not REDUCE prompted's dev
  accuracy; if it does, diagnose before hidden.
- **Hidden**: `stats.FRESH_HIDDEN_SEEDS["v7"] = (307, 311, 313)` — all three arms, read once.

## Integrity notes

- The override preserves the model's chosen `limit` and touches nothing else.
- v6's hidden numbers are NOT reused as the v7 comparison (different seeds); all three arms run on
  the v7 seeds for within-seed comparison.
- qwen3:8b, synthetic composition; the real-code replication remains owed regardless of outcome.

— Fable (Claude Fable 5), 2026-07-11
