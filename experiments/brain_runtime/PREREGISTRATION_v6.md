# Pre-Registration v6: Multi-hop composition — does forced structure help *thinking*?

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: Registered (no v6 model run yet)

## Motivation

Every prior task was single-hop and saturated at 0/1 (audit finding), so no experiment could
distinguish "the kernel improves reasoning" from "the kernel fixes plumbing." HYP-27 showed a
strong prompt matches the kernel on single-hop; the user's malform critique sharpened the open
question: **is there any regime where deterministic structure improves the *quality* of the
model's reasoning, not just the validity of its actions?** Composition is that candidate regime:
`outer(inner(a,b), c)` needs two retrievals-worth of rules surfaced and *chained* — a working-memory
discipline that structure might genuinely help.

## Task

`multihop_eval.py`: 12 operators, 60 depth-2 composition problems per seed; recurrence = both
operators already learned; answers graded from the last REASON output; incomplete = wrong.
`max_internal_actions=4` (the post-audit budget; 2 was shown to be a confound).

## Arms (the two that still matter after HYP-27/29)

- **`prompted`**: encoder retrieval + STRONG prompt, NO invariants (best-effort prompting).
- **`full_kernel`**: encoder retrieval + constrained conditional-schema actions + ordered
  invariants (retrieve-before-terminal, reason-after-last-retrieve — the HYP-29-fixed form).

## Hypotheses (pre-committed, symmetric)

- **H-v6a (structure helps thinking)**: on hidden seeds, `full_kernel` recurrence − `prompted`
  recurrence ≥ **0.15** AND `gap_significant` (Wilson, stats.py) — forced ordered structure
  measurably improves multi-hop reasoning beyond what prompting achieves.
- **H-v6b (prompting still suffices, equally reportable)**: gap < 0.15 or not significant — the
  kernel's value remains guarantees-only even under composition; the "improves thinking" claim
  is retired for this model class.
- **Saturation check**: if BOTH arms are ≥0.95 or ≤0.05 on dev, the task failed to create dynamic
  range — STOP, redesign (harder composition), do not read hidden.

## Splits

- **Dev**: seed 20260711 (sanity + saturation check only).
- **Hidden**: `stats.FRESH_HIDDEN_SEEDS["v6"] = (211, 223, 227)` — never used anywhere before.

## Model

qwen3:8b (the capable-model regime where HYP-27 showed prompting suffices on single-hop — the
strongest test of whether composition changes that).

## Integrity notes

- Same encoder store, same learning protocol, same budget for both arms; only the deterministic
  guarantee differs.
- Both arms' gates use Wilson significance (audit fix), not point estimates.

— Fable (Claude Fable 5), 2026-07-11
