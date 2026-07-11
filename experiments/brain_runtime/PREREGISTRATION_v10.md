# Pre-Registration v10: HYP-33 on REAL conversation texture (external validity for the ship)

Registered: 2026-07-11 by Fable (Claude Fable 5)
Status: 45%: Inconclusive-by-construction (stop-rule). v10.1 (30%): **NULL** — raw 0.900 vs attention 1.000, gap +0.100 n.s. Attention never lost a case; effect is geometry-scoped. See HYP-34.
Goal link: Goal v0.1 item (3) — the attention stage ships with real-texture evidence, not only
synthetic chatter.

## Question

HYP-33's +0.650 used synthetic small-talk as the distractor bed. Does the effect survive REAL
conversation texture — messy, topically overlapping, variable-length turns?

## Design (semi-synthetic: real bed, controlled ground truth)

Distractor bed = contiguous windows of REAL messages from this project's own agent-bus channel
(`agent_bus/bus.jsonl`, ~90 genuine multi-agent messages; roles mapped to user/assistant by
sender). Planted content (target fact + 3 same-attribute distractors + 1 forbidden value, same
templates and positions policy as v9) is INSERTED at seeded positions, replacing nothing — the
real messages are untouched distractors. This keeps ground truth controlled (the scorer's
expected/expected_source/forbidden are ours) while the retention/organization challenge is real:
bus messages are dense with technical content that embeds NEAR our planted attribute templates
(ports, budgets, seeds, task ids) — a much harder near-duplicate field than v9's small talk.

Window: 34 consecutive real messages + 1 system + 5 planted + 1 final query = 41 messages;
budget_chars = 45% of transcript (v9 policy). Windows chosen by seeded offset into the bus log;
the bus log is FROZEN at the commit registering this document (later messages excluded).

## Arms / metrics / model

Exactly v9: `raw` vs `attention`, `context_organization_eval.py` unchanged, qwen3:8b.

## Gate (pre-committed)

- **Confirmed** iff on hidden seeds `FRESH_HIDDEN_SEEDS["v10"] = (601, 607, 613)` (20 cases each):
  attention − raw success ≥ 0.15 AND Wilson-significant AND attention leak ≤ raw leak AND
  attention attribution ≥ raw attribution.
- **Partial / Null equally reportable**; if raw success ≥ 0.85 on dev (real texture might truncate
  less catastrophically), record Inconclusive-by-construction and report without a hidden read.
- Mechanism metric (`expected_present`) required, as in v9: retention vs positional.

## Dev

Seed 20260711, 20 cases, stop-rules as in v9.

— Fable (Claude Fable 5), 2026-07-11

---

## Dev outcome + Amendment v10.1 (2026-07-11, dated)

Dev (45% budget, real bed): raw success **0.95** (retention 1.00 BOTH arms) — the stop-rule fired:
real bus messages are long relative to the short planted facts, so 45% char-budget truncation did
not evict the target. Recorded as **Inconclusive-by-construction at 45%**; no hidden read at this
setting. Honest note for the ship: HYP-33's effect size is budget-geometry-dependent — chronology
only fails when pressure actually evicts mid-context facts.

**Amendment v10.1 (frozen before any further run):** identical design with `budget_fraction=0.30`
(the only change; generator parameterized). Dev sanity first with the same stop-rules; hidden
(601/607/613) only if raw < 0.85 on dev. — Fable (Claude Fable 5)
