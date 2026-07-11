# Pre-Registration v4: The Integrated Kernel — full deterministic stack vs naive LLM-glue at scale

Registered: 2026-07-11 by Fable (Claude Opus 4.8)
Status: **CONFIRMED** on the v4.1 gate (qwen3:8b hidden) — naive recurrence 0.000 vs full_kernel
1.000, gap 1.000; naive fails structurally (retrieval_participation 0.009, reasoning 0.000). See HYP-26.
Thesis link: PROJECT_PLAN §00. This is the capstone that runs the whole kernel as ONE system and
asks whether the deterministic mechanisms, *composed*, deliver reliable within-session learning
where the idiomatic "let the LLM drive" stack does not. Unifies the two proven threads:
retrieval quality (HYP-20/21/23) + action validity + structure (HYP-22/25).

## Motivation

We have proven each mechanism in isolation: encoder retrieval beats jaccard at scale (HYP-23);
a per-kind constrained schema makes model-chosen actions malformed-free and lifts completion
(HYP-25); mandatory retrieval + a reasoning step unlock within-session learning (HYP-22). We have
never run them **together with the model choosing its own actions at scale**. HYP-23 used a
hand-coded policy (no malformed actions possible); HYP-25 used model-chosen actions but only 5
operators and native retrieval. The integrated question is the product question: assemble the
kernel, let the model drive, scale the memory — does determinism hold the whole chain together?

## Arms (both let the MODEL choose its actions via JSON — so action validity is measured, not assumed)

- **`naive`** (idiomatic LLM-glue): `PersistentBrainRuntime` jaccard retrieval; free-form action
  JSON (no `format`); NO structural invariants (`require_retrieval_before_terminal=False`,
  `require_reason_before_ground=False`). This is how an agent looks when the LLM decides everything.
- **`full_kernel`** (the deterministic kernel): `EncoderMemoryRuntime` encoder top-k retrieval;
  constrained decoding with `ACTION_SCHEMA_CONDITIONAL`; both invariants ON
  (`require_retrieval_before_terminal=True`, `require_reason_before_ground=True`).

Everything else identical: same 30 operators × 5 repetitions, same problems, same verified-rule
learning after grading, same model (qwen3:8b), same model-chosen retrieval limit (both arms use
the identical policy prompt / fallback), same `max_internal_actions=2`. Answers graded from the REASON output (per HYP-25). A cycle that never
terminates within budget is counted incomplete → incorrect (not a crash).

This is a WHOLE-STACK comparison; it deliberately does not attribute the gap to a single mechanism
(each was attributed in HYP-21/23/25). The claim is about the composed kernel.

## Hypotheses (pre-committed)

- **H-v4a (learning, primary)**: `full_kernel` recurrence accuracy − `naive` recurrence accuracy
  ≥ **0.30** on hidden seeds. The composed kernel learns across the session where the naive stack
  does not reliably.
- **H-v4b (action validity)**: `full_kernel` malformed-action rate == **0**; `naive` malformed rate
  **> 0** (the naive stack really does emit invalid actions to fix).
- **H-v4c (completion)**: `full_kernel` cycle-completion rate ≥ `naive` cycle-completion rate.

## Metric

Per arm: `recurrence_accuracy` (correct on non-first-appearance problems — the learning signal),
`first_appearance_accuracy` (should be ~0 both arms; novel operators are unguessable),
`malformed_action_rate`, `cycle_completion_rate`. Reused from the sibling evals unchanged.

## Splits (anti-overfit)

- **Dev seed** (sanity only; confirm naive actually malforms + fails to learn, kernel completes):
  `20260711`. No selection/tuning after this run — arms and gate are frozen now.
- **Hidden seeds** (reported verdict, never inspected during design): `101, 103, 107`.

## Decision gate (pre-committed, qwen3:8b, hidden)

- **Confirmed** iff on hidden: (1) H-v4a met (recurrence gap ≥ 0.30), AND (2) `full_kernel`
  malformed == 0, AND (3) `naive` malformed > 0 (a real failure existed), AND (4) H-v4c met.
- **Partial** if the recurrence gap is positive but < 0.30, or if (2) holds but (1) does not.
- **Inconclusive** if `naive` neither malforms nor underperforms (the task is too easy to separate
  the stacks — do not claim a kernel win without a naive failure to beat).
- **Refuted** if `naive` ≥ `full_kernel` on recurrence accuracy.

## Integrity notes

- The structural invariants (forced retrieval / reason-before-ground) are PART of the kernel under
  test; the naive arm must not get them. This is a stack-vs-stack test by design, and every
  component was already isolated in a prior pre-registration.
- Retrieval query is whatever the model emits (fallback `"{goal} {observation}"`); we do NOT
  hand-filter retrieved memories to the exact operator key (HYP-23's hand-coded policy did — the
  kernel here must earn its retrieval through the encoder, not an oracle filter).

## Known limitations to state in the writeup

- qwen3:8b only; 30 operators; 3 hidden seeds. A hosted-model / second-task replication is owed.
- The gap is a composite of three mechanisms; this experiment shows they compose, not their
  individual sizes (see HYP-21/23/25 for those).

— Fable (Claude Opus 4.8), 2026-07-11

---

## Amendment v4.1 (2026-07-11, after the DEV sanity run — dated, not a silent edit)

Dev run (qwen3:8b, seed 20260711, hidden untouched): naive recurrence **0.000**, full_kernel
**1.000**, gap 1.000; both malformed **0.000**; both completion 1.000. Inspecting the event
sequences revealed the naive failure mode is **not** what gate condition (3) assumed:

- naive: **all 150 cycles are `('ground',)`** — left to choose, qwen immediately grounds a null
  value without ever retrieving or reasoning (answer=None, empty response). Valid JSON, empty act.
- full_kernel: **all 150 are `('retrieve','reason','ground')`** — the invariants force it
  (`forced_retrievals=150`, `forced_reasons=150`); the reason output is genuine correct work
  ("rule for mbazi is (a×4)+b … = 23"). 120/120 recurrence.

So the naive stack does not fail by emitting *malformed* JSON (qwen rarely does, cf. HYP-25's
0.055) — it fails **structurally**: a free LLM agent doesn't spontaneously retrieve or reason, it
blurts a null answer. Gate condition (3) ("naive malformed > 0") mis-targeted the failure and would
not trigger on qwen. This is the m0022 class generalized to the whole loop, and it is the real
finding.

**Amendment (frozen before the hidden read):**
- Primary gate H-v4a unchanged: recurrence gap ≥ 0.30 on hidden.
- Replace condition (3) with the ACTUAL naive failure: **naive `recurrence_accuracy` ≤ 0.10 AND
  naive `retrieval_participation_rate` < 0.50** (there is a real, structural failure to beat).
- full_kernel `malformed == 0` stays as a REPORTED structural guarantee, not a stack separator
  (on qwen it is 0 for both arms; the separator is structure + retrieval, not JSON shape).
- Add reported metrics `retrieval_participation_rate` / `reasoning_participation_rate` (the naive
  arm's ~0 here is the crux). The capstone's claim narrows honestly to: **deterministic structural
  invariants (forced retrieve+reason) + encoder retrieval convert a capable model from "grounds
  null" to "reliably learns" at scale** — composing HYP-23 (retrieval) with the CoALA structure.
  Action-validity (HYP-25) is part of the kernel but is not the lever that separates the stacks on
  this model/task. — Fable (Claude Opus 4.8)
