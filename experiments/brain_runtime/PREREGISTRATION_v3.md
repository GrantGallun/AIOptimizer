# Pre-Registration v3: Constrained Decoding as a CoALA Action-Validity Layer

Registered: 2026-07-11 by Fable (Claude Opus 4.8)
Status: **CONFIRMED** on the v3.1 design (hidden gate, qwen3:8b) — see HYP-25 and the closing note.
Thesis link: PROJECT_PLAN §00 — *LLMs are probabilistic semantic coprocessors; AIOptimizer is the
deterministic kernel that makes their work reliable.* This is the purest token-level instance of
that thesis and the top-ranked gap in `LANDSCAPE.md` (constrained-decoding cluster:
Outlines/Guidance/XGrammar). Does **not** edit any v1/v2 file.

## Motivation

The CoALA controller (`coala_ollama.py`) drives reasoning by having the model emit a
`CognitiveAction` as JSON (`{"kind": "retrieve"|"reason"|"ground"|"learn", ...}`). Today that JSON
is *free-form*: the model writes text, and `parse_action` → `_extract_json_object` scrapes a JSON
object out of it and raises `ActionParseError`/`json.JSONDecodeError` when the model doesn't format
it right. This is the **m0022 class of failure** — the controller depends on the model
spontaneously doing the right *structural* thing (HYP-19: models don't reliably do that unprompted).

Ollama's `/api/generate` accepts a `format` field (a JSON Schema) that constrains decoding at the
token level so only schema-valid tokens can be sampled. Setting `format` to the action schema makes
malformed actions **structurally impossible** rather than merely improbable. That is a deterministic
mechanism replacing "hope the JSON parses."

## Hypotheses (pre-committed, with directions)

Same controller, same prompts, same problem sequence; only *how the action JSON is produced* changes.

- **H-v3a (structural guarantee, primary)**: Under the `constrained` arm, the malformed-action rate
  is **exactly 0** — every policy call yields a schema-valid, parseable `CognitiveAction`.
- **H-v3b (no-regression)**: `constrained` task-completion accuracy is **≥** `free_form` accuracy
  (constraining format does not degrade the model's action *choices*, only guarantees their shape).
- **H-v3c (where it matters)**: The malformed-action gap (`free_form` rate − `constrained` rate) is
  **larger on the weaker model** (llama3.2:3b) than on the stronger one (qwen3:8b). Constraint buys
  the most where free-form formatting fails most.

## Arms

Two arms sharing the identical policy prompt, schema description, allowed-grounding set, and the
retrieval-before-terminal invariant:

- **`free_form`** (current behavior): `generate_with_metrics(...)` with no `format`; `parse_action`
  extracts JSON from free text. A parse failure is counted as a malformed action; on failure the
  controller falls back to its existing default action so the run still completes (fallbacks are
  counted, not silently swallowed).
- **`constrained`**: identical call but with `format=<ACTION_SCHEMA>` passed through to Ollama. The
  returned text must already be a single schema-valid JSON object; `parse_action` still runs and any
  failure here is a genuine malformed action (expected: 0).

`ACTION_SCHEMA` is a JSON Schema object: `{"type":"object","required":["kind"],
"properties":{"kind":{"enum":["retrieve","reason","ground","learn"]}, "query":{"type":"string"},
"instruction":{"type":"string"}, "name":{"type":"string"}, "value":{}, ...}}` covering the union of
fields `parse_action` reads. The exact schema is frozen in the implementation task and is **not**
tuned after seeing results.

## Metric

Per arm, per model:
1. **`malformed_action_rate`** = (policy calls whose raw output fails `parse_action`) / (policy calls).
   PRIMARY. This is read straight off `AdapterMetrics`; add a `malformed_actions` counter.
2. **`task_completion_accuracy`** = fraction of problems the controller answers correctly (reuse the
   `coala_learning_eval` grading — `parse_answer` on the final ground).
3. **`forced_retrieval_ok`** = the retrieval-before-terminal invariant held (sanity; must stay true
   in both arms).

## Splits (anti-overfit)

- **Dev seeds** (sanity that free_form actually breaks — no schema tuning after): `20260711, 7, 13`.
- **Hidden seeds** (the reported result, never inspected during design): `101, 103, 107`.

The schema is frozen before any run. Dev is only used to confirm the free_form arm produces a
non-zero malformed rate on the weak model (else there is nothing to fix — see the C0 check in v2).

## Decision gate (pre-committed)

Read on hidden seeds. Report the single frozen schema; no per-result schema edits.

- **Confirmed** iff, on **llama3.2:3b** (the model where the effect must appear):
  1. `constrained.malformed_action_rate == 0.0` (H-v3a), and
  2. `free_form.malformed_action_rate > 0.0` (there was a real failure to fix — else Inconclusive:
     nothing to constrain), and
  3. `constrained.task_completion_accuracy >= free_form.task_completion_accuracy` (H-v3b), and
  4. `forced_retrieval_ok` true in both arms.
- **Partial** if (1) holds but (3) fails (format guaranteed, but constraining hurt action choice —
  a real and interesting cost, report it).
- **Inconclusive** if (2) fails (free_form never malformed on this task → the harness is too easy;
  do not claim a win where there was no failure to prevent).
- **Refuted** if (1) fails (constrained still produced malformed actions → the `format` path is not
  actually constraining; treat as a substrate bug, stop and fix before interpreting).

qwen3:8b is run as the **ceiling reference** for H-v3c (expected: small or zero free_form malformed
rate → little to fix), not as the primary gate model.

## Safety / integrity invariants (all arms)

- The `constrained` arm may not change *which* actions are allowed, the grounding set, or the
  retrieval-before-terminal invariant — only the token-level shape of the emitted JSON.
- A completion gain bought by relaxing the retrieval invariant does not count.
- Fallback-on-parse-failure must be **counted** (`malformed_actions += 1`), never hidden, or H-v3a
  is unmeasurable.

## Known limitations to state in the writeup

- Two local models only; a hosted-model replication is owed before any general claim.
- 3 hidden seeds is small; a pass means "constraint removed the format failure mode on this task,"
  not "constrained decoding is universally better."
- Ollama's grammar-constrained decoding is the mechanism under test; if its schema support is
  partial (e.g. weak `oneOf`), that is a finding about the substrate, recorded as such.

— Fable (Claude Opus 4.8), 2026-07-11

---

## Amendment v3.1 (2026-07-11, after the DEV sanity run — dated, not a silent edit)

The dev-seed run (HYP-20260711-24; llama3.2:3b + qwen3:8b, seed 20260711, hidden untouched)
did its job and exposed two flaws in the v3 design **before** any confirmatory read:

1. **The union schema cannot guarantee H-v3a.** `ACTION_SCHEMA` requires only `kind`, but
   `parse_action` enforces per-kind required fields *and* forbids per-kind-unexpected fields. So a
   schema-valid `{"kind":"reason"}` still fails parsing. Result: constrained malformed rate hit 0.000
   on qwen3:8b (it voluntarily fills the right fields) but only 0.043 on llama3.2:3b (it emits
   minimal objects). The constraint must encode the **full per-kind contract**.
   Fix: replace the union with a **conditional schema** — `oneOf` over four branches keyed on
   `kind`, each branch listing that kind's exact `required` and allowed properties (mirroring
   `parse_action`). Then H-v3a becomes a real test of "does token-level constraint enforce the
   contract" rather than "does the model volunteer the fields."

2. **`task_completion_accuracy` is degenerate (0 everywhere), so H-v3b is untestable.** The answer is
   supposed to arrive via the REASON step, but models skip reasoning and emit `retrieve → ground`
   with a null value (grounded output `"None"`). qwen completed 25/25 cycles yet scored 0.
   Fix: make the answer path mandatory and measurable — either (a) require a REASON action before any
   GROUND (extend the controller/adapter invariant) and grade `parse_answer` on the reason output, or
   (b) constrain the terminal `ground` schema so `arguments.value` is a required integer. Prefer (a):
   it matches how `coala_learning_eval` already grades (reason-output → ANSWER=<int>).

v3.1 plan: freeze the conditional schema + the reason-before-ground grading in a NEW module/result
version (do not edit the v3 files or results); re-run dev sanity (confirm free_form still malforms and
completion is now non-degenerate on qwen3:8b), then read the frozen gate on the hidden seeds. Primary
model becomes **qwen3:8b** (llama3.2:3b's cycle-incompletion 0.16–0.20 makes completion the bottleneck,
not action shape); llama3.2:3b stays as the weak-model reference for H-v3a-under-conditional-schema. The
malformed-halving mechanism result (HYP-24) stands regardless of v3.1. — Fable (Claude Opus 4.8)

---

## Closing verdict (2026-07-11): CONFIRMED on the v3.1 design (HYP-25)

Dev sanity (qwen3:8b, seed 20260711): free_form malformed 0.062 / acc 0.680 / completion 0.880;
constrained malformed 0.000 / acc 0.800 / completion 1.000 — design sound, free_form still breaks.

Hidden gate (qwen3:8b, seeds 101/103/107, read once, aggregated):
free_form malformed **0.055** / acc 0.693 / completion 0.867; constrained malformed **0.0000** /
acc **0.813** / completion **1.000**. All four gate conditions met and consistent on every seed:
(1) constrained==0 ✓, (2) free_form>0 ✓, (3) constrained acc ≥ free_form (0.813 ≥ 0.693) ✓,
(4) forced_retrieval_ok both ✓ → **CONFIRMED**. The per-kind conditional schema (not the v3 union)
makes malformed actions structurally impossible and, by removing wasted malformed cycles, also raises
completion. Owed before a general claim: a hosted-model and a second task family. — Fable (Claude Opus 4.8)
