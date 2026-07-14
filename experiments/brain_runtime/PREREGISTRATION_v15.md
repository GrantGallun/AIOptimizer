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

---

## Harness frozen (2026-07-13) — scorer + fixtures live; runner-wiring delegated

Fable froze the objective scorer (`score_v15.py`, arm-blind: identical deterministic checks per
arm dir — file presence + requirement contracts via the shipped `aioptimizer.requirements` engine +
light Python AST checks; task passes iff files present AND functional contracts pass AND AST checks
pass; CONSTRAINT-tagged contracts feed a separate constraint-retention metric). Fixture schema +
the first 3 tasks are frozen in `v15_build_tasks.json` (self-tested: passes a constraint-honoring
artifact, fails a violating one, retention 0/3). Codex's paired workspaces (v15-ab-v4, plugin
A-on/B-off, byte-identical seeds) are ready.

Amendment: n starts at the 3 frozen tasks and grows to 12 by authoring MORE tasks to the same
schema (Codex, t0048) — the schema/scorer/gate are frozen now; adding tasks that fit the schema is
mechanical, not a research-design change. Gate unchanged (A−B pass ≥0.15 Wilson-significant AND
constraint-retention strictly better; null equally reportable). — Fable (Claude Fable 5)

---

## Pilot result (2026-07-14, dev-tier — NOT the pre-registered verdict)

gpt-5.6-luna, **1 paired replicate** (< the pre-registered >=2), 12 tasks. Result
`v15_ab_pilot_luna_r1_v4.json` (SHA256 1EDECD70...50; independent rescore identical):
- **Dead null on quality**: A (optimizer ON) 8/12 vs B (OFF) 8/12, gap 0.00, identical CIs
  [0.39,0.86], constraint-retention 0.84 both, **zero tasks flipped** (A_only=B_only=[]); both
  arms failed the SAME four (robots_aware_crawler, archive_cli_dry_run, env_token_config,
  scraper_structured_logging) — i.e. task difficulty, not treatment.
- **Cost-negative**: A used +16.9M input tokens and +767s vs B (injected context adds per-turn tokens).

**Interpretation (Fable):** Inconclusive-because-underpowered (n=1, wide CIs) WITH a diagnosed
design flaw, and the null trend is consistent with HYP-34. The tasks are short (8 turns) and
RE-STATE their constraints ("Reminder: honor the constraints"), which is the low-context-pressure
regime where the optimizer's advantage was already shown to collapse to ~0 (HYP-33's benefit needs
a constraint stated ONCE, early, then BURIED). So this pilot tested the wrong regime; it does not
refute the product, it mis-specified the condition. No HYP verdict written from a pilot.

## Amendment v15.1 (frozen before any rerun): make the tasks create real context pressure

Redesign the task scripts: (a) 24-40 turns each; (b) every hard constraint stated EXACTLY ONCE in
turn 1-2 and NEVER repeated (delete all "Reminder" prompts); (c) >=15 turns of substantive
intervening build work between a constraint and the turn that depends on it (bury it past the
model's easy-recall window); (d) keep the same objective scorer/gate unchanged. THEN run >=2
replicates for the confirmatory verdict. Rationale: only under this pressure can injected context
plausibly help; a null there would be a real refutation, a win there a real product result.
— Fable (Claude Fable 5)
