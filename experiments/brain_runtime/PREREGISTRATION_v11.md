# Pre-Registration v11: Usage-heat protection for foundational context (the user's compaction idea)

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: Registered (no v11 run yet)

## The idea (user, 2026-07-11) and the blind spot it targets

"If a conversation repeatedly uses a certain piece of context, that section should selectively not
get compacted." Query-relevance selection/organization (HYP-20/21/33) protects what resembles the
CURRENT query; a foundational paragraph the whole conversation leans on — but the latest query
does not lexically resemble — gets evicted. Heat = accumulated usage is the temporal prior that
fixes this (chunk-level analogue of H2O heavy-hitter retention).

## Task design (frozen): the codename-mapping test

Each case (synthetic bed, v9 geometry, budget 32% — must evict the mapping under query-only ranking):
- **Foundational turn** (early, T0003–T0006): a codename mapping — "Project <codename> refers to
  the <service> service." `expected` = the service word; `expected_source` = this turn.
- **4 usage turns** (middle): discuss events about "<codename>" (an outage, a deploy, a review,
  a migration) WITHOUT restating the mapping — these generate heat for the mapping turn via shared
  codename vocabulary.
- Distractors: v9-style chatter + 2 OTHER codename mappings (never used again — cold foundations,
  the control: heat must protect the USED mapping, not all mappings) + 1 forbidden value.
- **Final query**: "Which service had the <event>?" — answerable ONLY via mapping + usage turns;
  lexically dissimilar to the mapping turn itself.

## Arms

- `attention` (heat_weight = 0): the gate-passed HYP-33 mechanism — predicted to drop the mapping.
- `attention_heat` (heat_weight = **0.5**, frozen): t0033's heat-weighted ranking; heat computed
  from all user turns EXCEPT the final query (using it would smuggle current-query relevance in).
- `raw` chronology as reference.

## Gate (hidden = FRESH_HIDDEN_SEEDS["v11"] = (701, 709, 719), 20 cases each, qwen3:8b)

- **Confirmed** iff attention_heat − attention success ≥ 0.15 AND Wilson-significant AND
  attention_heat leak ≤ attention leak AND the COLD mappings' retention does not exceed the used
  mapping's (heat must be selective, or it is just "keep all foundations").
- Dev stop-rules (seed 20260711): attention (no heat) success < 0.85 — the design must actually
  evict the mapping under query-only ranking, else Inconclusive-by-construction (tighten budget in
  a dated amendment). Not both arms at floor/ceiling.
- Null equally reportable: if query-relevance already keeps the mapping (codename overlap may be
  enough), heat is unnecessary on this task — scope honestly.

— Fable (Claude Fable 5), 2026-07-11

---

## Dev-stage design log (2026-07-11, all render-only — zero model calls, dated)

Built: t0033 heat API (Codex), v11 runner (`heat_eval.py`), frozen dev cases. Render-level
iteration then killed four naive heat designs in sequence, each with a diagnosed mechanism:
1. Dense-cosine heat, user-turn queries: user records SELF-MATCH at cosine 1.0 → heat degenerates
   to "keep the user's own chatter" (mapping retention 0.05).
2. + self-exclusion: uniform chatter templates cross-heat each other (mapping rank 28/40) —
   fixed by diversifying the bed (a case-design artifact worth keeping fixed).
3. All-turns dense heat: topical recurrence heats everything (0.05).
4. IDF-lexical reference heat (rare-token overlap): signal improves (mapping heat rank ~12/40)
   but the CLUSTER-BLEND semantics still displace the mapping relative to plain relevance
   (0.40 vs 0.65 at a well-posed 45% budget; the earlier 32% budget made the task unanswerable
   by construction — pinned + usage turns already exhausted it).

**Standing conclusion (dev-stage):** heat as a *ranking blend* competes with relevance instead of
complementing it. The user's original phrasing — "keep the entire thing" — points at PROTECTED-SET
semantics (top-K heat records pinned like system/query, relevance ranks the remainder), likely with
reference-graph heat (explicit citation/anaphora) rather than similarity accumulation. That is the
v11.3 design to attempt fresh — no model run is spent until a render check passes
(attention_heat expected_present > attention's). Gate unchanged. — Fable (Claude Fable 5)

## v11.3 outcome (2026-07-11, render-only, dated) — naive heat is dev-stage REFUTED as implemented

Design 5 (introduction-reference heat: idf × later-use of first-introduced terms, PROTECTED-SET
pinning per the user's original phrasing): mapping reaches heat rank 2–4, but early chatter turns
introduce most of the conversation's vocabulary and out-rank it (protected-set K=2 misses the
mapping in 3 of 4 inspected cases; retention 0.45 vs plain attention 0.65).

**Standing verdict (dev-stage, no model calls spent):** five unsupervised heat formulations
(dense-cosine ×3 variants, flat IDF overlap, introduction-reference; blend AND protected-set
semantics) each fail the render check for a distinct diagnosed reason. On a 40-turn bed, lexical/
embedding "usage" signals cannot reliably separate a 4-reference foundation from ordinary
vocabulary recurrence. The idea's surviving implementations, for a future prereg:
(a) **explicit-citation heat** — count actual references (our T####/attribution machinery; agent
conversations genuinely cite task ids/commits), ground truth instead of lexical guessing;
(b) much longer/realer beds where genuine usage dominates recurrence noise.
The v11 hidden seeds (701/709/719) remain UNREAD and reserved. — Fable (Claude Fable 5)
