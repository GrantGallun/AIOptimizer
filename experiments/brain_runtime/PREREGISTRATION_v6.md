# Pre-Registration v6: Multi-hop composition — does forced structure help *thinking*?

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: **H-v6b per the frozen gate** (hidden 211/223/227): prompted 0.884 vs full_kernel 0.987 — gap +0.103, Wilson-significant but below the 0.15 pre-committed effect size. See HYP-30.

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

---

## Amendment v6.1 (2026-07-11, after dev sanity — dated, not a silent edit)

Dev (qwen3:8b, seed 20260711): prompted 0.039 / full_kernel 0.137 — near-floor but the stop-rule
did not trigger (kernel > 0.05). Row inspection showed the floor is a MEASUREMENT artifact, not
reasoning failure: in most wrong rows the model retrieved both rules and computed the inner hop
correctly, then hit `max_reason_tokens=96` mid-second-hop; last-integer grading then read a
truncated thought. Fix (symmetric, both arms): `max_reason_tokens=256` for the multihop eval only.
No gate, arm, or seed changes. Dev rerun next; hidden (211/223/227) only after a valid dev.
Observed-but-not-relied-on: kernel led 0.137 vs 0.039 under truncation (its 100% completion and
retrieval breadth) — this is NOT evidence for H-v6a; the gate reads only the post-fix hidden run.
— Fable (Claude Fable 5)

---

## Amendment v6.2 (2026-07-11, after the v6.1 dev rerun — dated, not a silent edit)

v6.1 dev (256 reason tokens): prompted **0.118** / full_kernel **1.000**. Row inspection: 31/39 of
prompted's wrong completed rows cite a MISSING rule — the v4.2 strong prompt's query template
("<operator name> rule", singular) makes the model retrieve one of the two needed rules, while the
kernel's deterministic fallback query carries the full observation and surfaces both. That is half
genuine mechanism (deterministic query construction IS a kernel feature) and half stale-prompt
artifact (best-effort prompting for THIS task would say "retrieve every operator's rules"). Per the
same fairness standard as HYP-26→27, the prompted arm gets a composition-aware strong prompt
(`STRONG_POLICY_SYSTEM_MULTIHOP`: retrieve ALL named operators' rules in one query, inner-then-outer
reasoning). Gate, arms, seeds unchanged. Kernel's 1.000 dev stands (its arm is untouched). Dev rerun
of the prompted arm, then hidden (211/223/227) for both arms. If prompted recovers → prompting-
suffices extends to multi-hop (H-v6b); if it still lags significantly → H-v6a with a clean
conscience. — Fable (Claude Fable 5)
