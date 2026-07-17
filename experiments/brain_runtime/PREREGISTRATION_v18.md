# Pre-Registration v18: Does the product beat NO product? (untreated baselines)

Registered: 2026-07-17 by Fable (Fable 5) — **before any v18 run, dev or hidden.**
Status: REGISTERED — not yet run.

## Origin — the external-validity hole the 2026-07-17 audit exposed

Every strong result we have (HYP-38 cliff, HYP-39 cost dividend) compares attention ordering
against the compiler's OWN chronological mode under the harness's first-fit packer. The audit
proved that "raw" arm is not what an untreated agent experiences (its retention *rises* as budget
shrinks; 29/120 same-case flips impossible under tail truncation). **No experiment has ever run
a genuinely untreated transcript.** v18 closes that hole. This is the experiment that decides
whether the flagship claim survives outside our own harness.

## Design

Frozen `generate_volume_cases` (byte-identical generator), N=320, 40 cases/seed, qwen3:8b local
($0), deterministic scorer (`score_response`) and prompt scaffold (`prompt_for`) reused from the
frozen harness. **`num_ctx=16384` uniformly for every arm** (the plain transcript is ~25k chars
≈ ~7k tokens; Ollama's default 4096 would silently truncate the very arm under test — every arm
gets the same decode config, only context construction varies). Four arms:

1. **untreated_full** — the plain chronological transcript, complete, rendered as
   `role: content` lines. No compiler, no budget. *The "no product, window fits" baseline.*
2. **untreated_tail** — same plain rendering, oldest messages dropped at message boundaries
   until it fits **2600 chars** (the treated arm's budget). *The "no product, budget-truncating
   agent" baseline.*
3. **attention** — compiler + organized render @2600 (the treated arm, as in v16/v17).
4. **raw_compiler** — compiler chronological render @2600 (continuity arm linking v18 to
   v16/v17; the ordering control).

Hidden seeds v18 = **(1409, 1423, 1427)**, fresh, read ONCE. Dev seed 20260711.

## Pre-committed predictions

- **P1**: untreated_tail ≈ 0 success — tail truncation drops the first-15% fact, and success
  without the fact in context has never occurred (0/167 across v16+v17).
- **P2**: attention ≈ 1.0 (v17's level).
- **P3 (the genuinely uncertain one)**: untreated_full ≥ 0.9 — at only ~7k tokens, qwen3:8b
  retrieves a buried unique fact from a full window. I.e., **I predict NO quality win at this
  scale**; the honest product claim vs. doing-nothing would reduce to *equal quality at <25% of
  the input tokens*. If this prediction FAILS (untreated_full materially below attention), that
  is a quality win for the product under a full window — surprising and equally reportable.

## Gates (committed before any run)

- **H-v18a (flagship external validity)**: CONFIRMED iff
  `wilson_lower(attention) > wilson_upper(untreated_tail)` on pooled hidden. Informative:
  both arms spend the same budget.
- **H-v18b (vs. full-window no-product)** — pre-committed three-way reading, no post-hoc
  reinterpretation:
  - CIs of attention and untreated_full **overlap** AND `wilson_lower(untreated_full) >= 0.9`
    → verdict "**parity**: same quality at <25% of the tokens vs. doing nothing."
  - `wilson_lower(attention) > wilson_upper(untreated_full)` → "**quality win** even when the
    window fits."
  - `wilson_upper(attention) < wilson_lower(untreated_full)` → "**the product HURTS** vs. doing
    nothing when the window fits" — the flagship claim is buried for fits-in-window regimes.
- **Token accounting** is reported per arm, but (v17 amendment lesson) the attention-vs-full
  token ratio is true BY CONSTRUCTION and is **not** a gate conjunct. The evidential content of
  v18 is the quality comparison; the token ratio only becomes a *claim* when quality parity or
  better licenses it.

## Dev sanity (seed 20260711 — STOP on any failure, no hidden read)

1. Mechanical, no model: untreated_full context contains the fact in 100% of cases (by
   construction); untreated_tail LACKS the fact in ≥90% (geometry check — if the tail retains
   the fact, the arm does not mean what it claims); attention fact-presence ≈ 1.0 (v17 level).
2. Model: attention success ≥ 0.9 at budget 2600 (v17 dev reproduced 0.975).

## Amendment v18.1 (2026-07-17, at dev stage, BEFORE any hidden read — endpoint corrected)

Dev exposed an instrumentation flaw in the committed scorer: `score_response`'s `success`
requires naming the source turn label (`T0013`), and those labels exist ONLY in the compiler's
rendering. A plain transcript structurally cannot cite them — untreated_full scored 0.025
"success" while being **40/40 correct on the value**, failing only the attribution field it was
never allowed to see. Reusing that endpoint would have rigged v18 for the product.

Corrected, committed before hidden:
- **Primary endpoint (all gates): `answer_correct`** — value retrieved, no forbidden leak.
  Fair across all four arms.
- `success` (value + attribution) is reported as a SECONDARY, clearly-labeled product-capability
  metric — provenance is a real feature the compiler adds, but it is a capability delta, not a
  quality comparison, and it carries no gate weight.
- H-v18a and the H-v18b three-way read are unchanged in structure, evaluated on `answer_correct`.
- v16/v17 are unaffected: both of their arms carried compiler labels, so `success` was fair
  within those designs.

## Scope (deliberate non-claims)

qwen3:8b + synthetic single-fact retrieval + N=320 (~25k chars ≈ 7k tokens) only. The
"window fits" regime tested here is small by frontier standards; v18 licenses NO claim about
100k+ contexts, agent builds, or frontier models. The encoder/compile cost remains unmeasured.

— Fable (Fable 5), 2026-07-17
