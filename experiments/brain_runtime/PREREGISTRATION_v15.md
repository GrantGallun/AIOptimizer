# Pre-Registration v15: The dogfood A/B — does the optimizer make the agent BUILD better?

Registered: 2026-07-12 by Fable (Claude Fable 5)
Status: Design frozen; fixtures/harness pending (t0044); runs after v14 lands.
Provenance: the user's idea ("build a series of prompts and see if with the ai optimizer it
builds it any bit better"). This is the product claim measured directly on real agent work.

## Design: paired scripted multi-prompt builds

- **Tasks**: 12 scripted build sequences (~8 prompts each) in the TalentTrader domain and similar
  (scraper/reader/CLI utilities). Early prompts establish CONSTRAINTS and FACTS ("never emit
  approval messages", "the output schema is X", "rate-limit at Y"); later prompts require them
  without restating. Each task ships with OBJECTIVE scoring: fixed tests + AST checks (v13
  software-probe style) + requirement contracts (must_include/must_exclude on the final artifact).
- **Arms**: identical Codex version/model, fresh workspace + fresh thread per run; A = plugin ON
  (gateway + injection), B = plugin OFF. Same prompt script verbatim.
- **Metrics**: per-task pass (tests + contracts), constraint-retention rate, tokens, wall time.
  Wilson intervals; paired per-task comparison. n=12 tasks × ≥2 replicates per arm.
- **Honest hazards, pre-committed**: agent nondeterminism (replicates + paired design); tiny n
  (report CIs, no overclaiming); the scorer must never see arm identity (files scored by path-
  anonymized copies).

## Gate

Confirmed iff arm A − arm B task-pass ≥ 0.15 AND Wilson-significant AND constraint-retention
strictly better; Null (equally reportable): the optimizer does not measurably change real build
quality at this n — receipts/token savings remain its case.

— Fable (Claude Fable 5), 2026-07-12
