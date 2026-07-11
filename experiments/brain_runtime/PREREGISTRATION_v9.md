# Pre-Registration v9: Attention-organized context vs raw chronology at matched budget

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: **CONFIRMED** (hidden 503/509/521): raw 0.350 vs attention 1.000 — gap +0.650, Wilson-significant, attribution 60/60, zero leaks. Mechanism = RETENTION. See HYP-33.
Origin: the user's context-compiler idea (Codex thread `context-compiler`). Codex built the
substrate — `ConversationCompiler` (pairwise-cosine clusters ranked against the active query,
system + latest-user pinned, provenance retained), matched-budget renderers, and a neutral
external-frozen-case evaluator (`context_organization_eval.py`) — and correctly stopped at the
research boundary. This document freezes the cases and the gate.

## Question

At an IDENTICAL character budget, does attention-organized context (semantic clusters ranked by
query relevance) beat raw chronological truncation on answer accuracy + source attribution — or is
chronology already fine? This extends HYP-20/21 from *selection* to *organization* and probes
lost-in-the-middle positionally.

## Case generation (frozen; generator implementation = Codex t0029, parameters here are binding)

Deterministic from a seed. Each case: a synthetic conversation of **40 turns** (T0001..T0040), each
turn one short user/assistant exchange on a distinct mundane topic (template pool ≥ 20 topics).
Planted content, positions uniform over T0005–T0030 (the "lost middle"):
- **1 target fact**: `"the <attribute> for <name> is <value>"` (varied templates; value = a
  4-digit number or 5-letter code unique in the transcript). `expected` = value,
  `expected_source` = the planting turn's T####.
- **3 near-topic distractors**: same attribute template, DIFFERENT names, different values
  (attribution stress — the scorer requires the right value AND the right source turn).
- **1 forbidden item**: a value explicitly marked `"private — never repeat"` in its turn;
  `forbidden` = [that value] (leak check).
- Final `query` asks for the target fact by name. `budget_chars` = **45%** of the full rendered
  transcript length (forces both arms to drop >half the material).

**Dev**: seed 20260711, 20 cases. **Hidden**: `FRESH_HIDDEN_SEEDS["v9"] = (503, 509, 521)`,
20 cases each (n=60/arm), qwen3:8b, read once.

## Arms (evaluator as built — no changes after this registration)

`raw` (chronological, recency-truncated to budget) vs `attention` (cluster-organized, same budget,
same pinning). Same model, system prompt, scorer.

## Gate (pre-committed, hidden)

- **Confirmed** iff on hidden: (1) attention − raw `success` ≥ **0.15** AND `gap_significant`
  (Wilson); (2) attention `leak_rate` ≤ raw `leak_rate`; (3) attention `source_correct` (among
  value-correct rows) ≥ raw's (organization must not break attribution).
- **Partial**: significant but < 0.15, or (1) holds while (3) narrowly fails — report the split.
- **Refuted/Null (equally reportable)**: no significant gap → chronology suffices at this budget;
  the middleware stays a config option, not a default.
- **Mechanism metric (required in the writeup)**: per-arm `expected_present` (did the target fact
  survive the budget?). If presence differs, the effect is RETENTION (selection redux); if presence
  is equal and success differs, the effect is POSITIONAL (true lost-in-the-middle organization) —
  distinguish honestly.

## Dev sanity stop-rules

Free of saturation (not both arms ≥0.95 or ≤0.05); raw must show a real deficiency to fix
(raw success < 0.85), else Inconclusive-by-construction — widen budget pressure before hidden.

— Fable (Claude Fable 5), 2026-07-11
