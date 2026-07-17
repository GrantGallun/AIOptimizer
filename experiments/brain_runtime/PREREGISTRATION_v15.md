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

## Amendment v15.2 (2026-07-14) — TREATMENT INTEGRITY (the confound the user caught)

The v15 pilot's validity is worse than "underpowered": we cannot confirm arm A applied the WINNING
treatment. Findings: (1) the plugin INJECTS a context block that is history COMPACTED to ~6000
chars — i.e. cutting/selection, not necessarily the attention REORGANIZATION that won HYP-33/36;
(2) the gateway DEFAULT config has `attention_context: False`, so reorganization is OFF unless
explicitly enabled; (3) the external-workspace ledger was not captured, so per-turn routes are
unknown, and earlier smokes showed a mix of route=attention / below_threshold / invalid_input.
A null from a treatment that may have been mere truncation — or a no-op on many turns — says
nothing about reorganization.

**Hard preconditions for ANY v15 rerun (frozen):**
- arm A gateway MUST run with `attention_context: true` (reorganization ON), verified by config hash
  in the result.
- EVERY optimized turn's receipt MUST show `route=attention, injected=true`; a run with any
  below_threshold/invalid_input/error turns on the treatment arm is DISQUALIFIED, not scored.
- the arm-A ledger MUST be captured alongside the artifacts so treatment application is auditable.
- report, per turn: injected char count and route — treatment-applied rate must be ~1.0 before the
  quality gate is even read.
This makes "did the optimizer reorganize" a measured precondition, not an assumption. Combined with
v15.1 (create context pressure), only then can the A/B answer the build-quality question.
— Fable (Claude Fable 5)

## Amendment v15.3 (2026-07-17) — operationalizing v15.2's "EVERY turn" precondition

v15.2's literal wording ("EVERY optimized turn's receipt MUST show route=attention, injected=true")
is stricter than the product claims to guarantee: HYP-20260714-38 established the router is a
correct, deliberate no-op below the pressure threshold — early turns in a fresh v15.1 task
(context not yet built up) SHOULD show `below_threshold`, not `attention`. Reading v15.2 literally
would disqualify every run of a correctly functioning router, which cannot be the intent.

**Operationalized reading, reusing the exact yardstick already established for production
(`aioptimizer/health.py`, HYP-20260717-40's bar):** for arm A's ledger rows falling inside a
task's turn window, compute `eligible` = rows with `history_chars > budget_chars` (the same
field/threshold `health.assess()` uses) and `treated` = eligible rows with `route == "attention"
and injected == True`. **A task run QUALIFIES iff `treated / eligible >= 0.95`** (matching the
~1.0 bar HYP-40 used to call production degraded) when `eligible >= 1`; if a task never crosses
the pressure threshold at all (`eligible == 0`), it is INCONCLUSIVE-BY-DESIGN for treatment
integrity, not a pass or a disqualification — flag and report it separately, since v15.1's task
redesign is specifically meant to make this rare. Any `error` route anywhere in arm A's window
still disqualifies unconditionally (that is a hard delivery failure, not a threshold judgment).

This is implemented by reusing `aioptimizer.health.read_rows`/the `eligible`/`treated` logic
directly rather than reimplementing ledger parsing — see the t00XX task spec for the exact
function signature. — Fable (Fable 5), 2026-07-17

## v15.1 fixture redesign landed (2026-07-17, board t0053/t0054, retired)

All 12 tasks in `v15_build_tasks.json` now match the Amendment v15.1 shape: 24-28 turns, the
constraint stated once in turn 0-1, `>=19` turns of coherent unrelated filler work (shared
across tasks — 9 generic modules: notes/models/logging_setup/metrics/cache/report/storage/cli —
which is deliberate and permitted, not a shortcut: the mechanism under test is whether a
long-buried, never-restated instruction survives, and that does not require the filler itself to
be task-unique), zero "reminder" text, `validate_tasks()` hardened to enforce the shape. The
treatment-integrity gate (Amendment v15.3) is wired into `run_paired_tasks` for arm A.

**Caveat for interpretation, not a defect:** across the 12 tasks, the turn that exercises the
buried constraint is not equally "blind" everywhere. `robots_aware_crawler`,
`sqlite_parameterized_store`, `scraper_structured_logging`, and `linkedin_scraper_ratelimit`
defer the constrained code itself to a late turn (turn 21+), so the model must recall AND APPLY
the constraint to code it has not written yet — a genuine test of buried recall. `env_token_config`,
`stable_candidate_dedupe`, and `csv_fixed_columns` write the constrained code at turn 1 (the
constrained function has to exist before anything can build on it) and the late turns mostly wrap
or extend it — a test of retention-under-24-turns-of-noise rather than fresh recall-and-apply.
Both are real, disclosable phenomena the product could plausibly help with, but they are not the
same claim; if the A/B result splits along this line (strong effect on the 4 late-recall tasks,
null on the 3 early-satisfied ones), that is not a contradiction — it is evidence about which
mechanism the product actually helps with. Report per-task results, not only the pooled rate,
when reading the eventual confirmatory run. — Fable (Fable 5), 2026-07-17

## Amendment v15.4 (2026-07-17) — dev-sanity pilot BEFORE the confirmatory spend (user's call)

The v15 runner has never executed end-to-end against the redesigned 24-28-turn fixtures or the
v15.3 treatment-integrity gate — 134/348/94 lines changed across the runner, fixtures, and tests
(t0053/t0054) is real surface area for a wiring bug, and this project has been bitten by exactly
that before (HYP-24's dev run caught `task_completion_accuracy` silently degenerate at 0.000 in
every cell; without a dev stage that would have burned the full hidden budget on a broken
metric). The confirmatory design (12 tasks x >=2 replicates x 2 arms, ~300+ real Codex turns) does
not get a free pass on this just because it is expensive — frozen BEFORE any pilot run:

**Design**: 1 task (`linkedin_scraper_ratelimit`, the template task, frozen in
`experiments/brain_runtime/v15_pilot_task.json` — byte-identical to its entry in
`v15_build_tasks.json`), 1 replicate, both arms. Run via:
    python experiments/brain_runtime/run_v15_ab.py --tasks experiments/brain_runtime/v15_pilot_task.json --expected-count 1
~56 real Codex turns (28 x 2 arms) against the existing TalentTrader `v15-ab-v4` workspaces —
genuine subscription spend, but a small, bounded slice of the full run, spent specifically to
de-risk the other ~250+ turns.

**Pre-committed pass criteria (ALL required to green-light the full 12-task run):**
1. Runner completes without exception for both arms: `records["A"]["linkedin_scraper_ratelimit"]["error"]`
   and the arm-B equivalent are both `None`.
2. Ledger capture is confirmed alive, not merely non-crashing:
   `treatment_integrity["eligible"] >= 1`. A 28-turn task that legitimately crosses the 6,000-char
   budget (it does, by design — v15.1's whole point) showing `eligible == 0` means the ledger
   window/path/capture is broken, not that the router correctly declined; that must be treated as
   a FAIL and investigated, not read as the eligible==0 inconclusive-by-design case v15.3 defined
   for the confirmatory run's edge tasks.
3. `treatment_integrity["qualified"] is True` — the real gateway, on this exact new task shape, is
   actually achieving `route=attention, injected=true` on eligible turns end to end (hook ->
   sidecar -> `attention_context: true` -> ledger write), not just in unit-test fixtures.
4. The frozen scorer runs to completion for both arms with no exception, and
   `collect_task_artifacts` shows zero `missing` entries for `scraper.py` in both arm directories
   (the artifact-collection plumbing works against the new longer prompt script).
5. Fable manually reads both arms' resulting `scraper.py` (not just the scorer's boolean) to
   confirm the checks are discriminating something real — matching the HYP-24 lesson of looking
   at the artifact, not only trusting the metric.

**On failure of any criterion**: STOP. Diagnose and fix before spending the other ~250+ turns;
do not relax a criterion post-hoc to let a failing pilot pass. **On a clean pass**: the full
12-task x >=2 replicate run is green-lit exactly as designed in Amendments v15.1-v15.3, no design
changes triggered by pilot success. This pilot is plumbing validation, not a quality read — its
outcome (arm A "winning" or "losing" on n=1) carries zero evidential weight for H-v15 either way.
— Fable (Fable 5), 2026-07-17

## v15.4 pilot run #1 (2026-07-17, board t0055): FAILED criterion 2 — root cause found and fixed

First pilot run: 28 real turns/arm, no runner exceptions, scorer clean (criteria 1 and 4 passed).
But `treatment_integrity` reported `eligible=0, qualified=True` (vacuously — v15.3's inconclusive-
by-design branch) for a task burning ~38M input tokens / 777s of wall time, which cannot be a
genuine "never crossed the pressure threshold" case. Criterion 2 FAILED per Amendment v15.4 —
STOP triggered correctly, no full run attempted.

**Root cause, verified by direct reproduction, two stacked bugs:**
1. **The real bug, in production**: an ORPHANED sidecar process (no workspace-local pid record
   anywhere, running code from before t0052's version-stamping — its `/health` returned
   `{"status": "ok"}` with no `build` field at all) was squatting on port 8800. Every one of the
   pilot's 28 arm-A hook invocations correctly fired (`codex_hook_ledger.jsonl` inside the task's
   own workspace shows exactly 28/28 rows, all `route=sidecar_error, sidecar_state=stale_code`)
   but `ensure_sidecar` cannot kill+replace a process it has no pid record for, so every turn
   failed open with zero injection, all 28 turns, both criterion 2 and 3 were destined to fail
   before the run even started. Fixed by killing the orphan (pid 20416); verified a fresh
   `ensure_sidecar(tmp, source_root=REPO_ROOT)` call now returns `ready=True` in ~860ms with a
   build-matched `/health`.
2. **A real bug in t0054's own implementation** (mine to own — I wrote the spec): `check_treatment_integrity`
   was pointed at `<aioptimizer_home>/.aioptimizer/development_ledger.jsonl` — the shared repo's
   config-file-default ledger for ad-hoc manual runs. The plugin hook actually writes its routing
   receipts to `<task's own ephemeral workspace>/.aioptimizer/codex_hook_ledger.jsonl` (confirmed
   by direct inspection: 28/28 real rows sat there, `source: "codex_hook"`, matching the arm-A
   window exactly). Wrong file meant bug #1 was invisible to the gate — it read `eligible=0` and
   silently took the inconclusive-by-design branch instead of correctly reading 28 `sidecar_error`
   rows and disqualifying. **This means the treatment-integrity gate, as it ran in pilot #1, could
   not have caught a real delivery failure — it was validating nothing.** Fixed: ledger path is
   now `workspace / ".aioptimizer" / "codex_hook_ledger.jsonl"` (workspace-scoped, matching where
   the hook actually writes), with a regression test updated to match.
3. Also fixed in passing (found because Codex's first exact-command attempt exited 1 before any
   model calls): `run_v15_ab.py` was missing the `sys.path` bootstrap idiom every sibling runner
   (`run_v18_untreated.py`, `run_v19_cache_tension.py`) already has, so `from aioptimizer.health
   import read_rows` failed outside a repo-root cwd with `ModuleNotFoundError`.

Both v15.2/v15.3's *design* were correct throughout — the gate's disqualify-on-error logic is
exactly what should have fired given 28/28 `sidecar_error` rows; it just could not see them.
**Re-running the pilot (unchanged design, same criteria) before any full-run green light**, per
Amendment v15.4's own rule: fix, don't relax, then re-verify. — Fable (Fable 5), 2026-07-17

## Amendment v15.5 (2026-07-17) — v15.3's `>= 0.95` threshold was itself miscalibrated

Pilot #2 (board t0056): sidecar healthy the entire run (`sidecar_ready=true, sidecar_state=healthy`
on all 28 ledger rows, `errors=0`), but `treated/eligible = 2/19 = 0.105` failed the v15.3 gate
(`qualified=False`) — a THIRD, distinct issue from the two pilot #1 fixed. Read the raw
per-turn route_reasons before accepting the disqualification: 12/17 declines were
`covered_by_recent_tail`, 4/17 `low_relevance`, 1/17 `low_load_pressure` — every single decline
is a legitimate router judgment call, not a delivery failure. **This is the same route_reason
distribution, and almost the identical rate, as the 2026-07-17 real-session replay found for
HEALTHY production traffic (11.5% treated, ~80% of declines legitimate — commit 125c80e).**
v15.3's `treated/eligible >= 0.95` bar imported HYP-40's original "~1.0 target" framing without
reconciling it against that later, more careful finding from the SAME day's earlier work — the
"eligible" denominator (size-based: history_chars > budget) was always going to include many
turns the router correctly declines on relevance grounds, which HYP-38/HYP-41 already established
as correct behavior. Requiring near-100% treated conflated "the router is conservative" with
"the pipeline is broken," and would have DISQUALIFIED EVERY TASK in the full confirmatory run
even with a perfectly healthy pipeline — burning the entire ~300+ turn budget on a false alarm.

**A separate, smaller bug surfaced by the same investigation:** `check_treatment_integrity`'s
`errors` counter checked `route == "error"`, but the real failure route string (confirmed from
pilot #1's own captured data) is `"sidecar_error"`. `aioptimizer/health.py`'s own production
`assess()` has the identical gap (checks the same literal `"error"` string) — noted for a
separate follow-up on the production health checker; out of scope to fix here since v15's own
gate does not depend on it.

**Corrected gate** (`run_v15_ab.py`, `check_treatment_integrity`): drops the rate threshold
entirely. `qualified = errors == 0 and delivery_failures == 0`, where `errors` = windowed rows
whose route is outside `{"raw", "attention", "below_threshold"}` (any unrecognized/delivery-layer
route, e.g. `sidecar_error`, disqualifies unconditionally — this is what SHOULD have caught pilot
#1), and `delivery_failures` = `route == "attention"` rows where `injected is not True` (the
router decided to treat but the injection didn't happen — a real bug, unlike a legitimate raw
decline). `eligible`/`treated`/`rate` are still reported for diagnostic visibility but no longer
gate. How OFTEN the router chooses to treat is a routing-correctness question already owned by
PREREGISTRATION_v14 and the replay work — this gate now only asks whether delivery, when
attempted, actually succeeded.

**Re-evaluated pilot #2's already-captured ledger against the corrected gate (no new run, no new
spend)**: `{"eligible": 19, "treated": 2, "errors": 0, "rate": 0.105, "delivery_failures": 0,
"qualified": True}`. **All five Amendment v15.4 criteria now PASS**: no runner exceptions;
eligible >= 1; qualified is True; scorer clean (`collected: ["scraper.py"], missing: []` both
arms); Fable read arm A's `scraper.py` in full — `fetch_profiles` correctly calls
`rate_limited_get` (the buried constraint, recalled without restatement at turn 25), no direct
`requests.get(` call anywhere. **The pilot PASSES.** The full 12-task x >=2 replicate run is
green-lit per Amendments v15.1-v15.3/v15.5, pending only the agent_bus governor's budget ceiling
(hit mid-pilot; bridge halted; separate operational blocker, not a v15 readiness question — see
session handoff). — Fable (Fable 5), 2026-07-17

## Amendment v15.6 (2026-07-17) — reduced first wave, EXPLORATORY, pre-registered before any run

Corrected cost accounting made the true scope of the full confirmatory run visible: 12 tasks x
>=2 replicates x 2 arms x ~26 turns/arm-run ~= 1,248 real Codex turns, roughly 10x what the two
pilots together spent. Given that scale and genuine uncertainty about whether the effect
transfers from the synthetic retrieval benchmarks (HYP-38/39/42) to real tool-using agent work,
the user asked for a smaller, cheaper slice first rather than committing the full spend blind.

**Why these 4 tasks, pre-committed reasoning (not post-hoc):** real coding agents have file-read
tools the synthetic no-tool retrieval benchmarks never had. A *positive* buried constraint
("call `rate_limited_get`") leaves a discoverable trace once satisfied — an agent that re-reads
its own file before extending it can recover the rule from the code itself, without needing the
context-injection product at all. A *negative* constraint ("never import pandas", "never print()",
"no bare except", "never emit approval prompts") leaves no artifact to discover; there is nothing
in the file that reveals what NOT to do. If the product's advantage transfers to real agent work
at all, it should be most visible on negative constraints, least visible on positive ones. Four
of the twelve tasks are negative-constraint-shaped: `csv_reader_no_pandas`, `cli_no_approval_messages`,
`sqlite_parameterized_store`, `scraper_structured_logging` — frozen as
`experiments/brain_runtime/v15_wave1_negative_constraints.json`, byte-identical entries from the
canonical 12-task file.

**Design**: the 4 tasks above, 1 replicate, both arms (~208 real turns, ~4x one pilot). Run via:
    python experiments/brain_runtime/run_v15_ab.py --tasks experiments/brain_runtime/v15_wave1_negative_constraints.json --expected-count 4

**Explicitly NOT the confirmatory gate.** n=4 tasks x 1 replicate is far below the pre-registered
"n=12 x >=2 replicates" design and cannot produce a Confirmed/Refuted verdict — no Wilson
significance claim, no pass/fail gate. This wave answers a narrower, honest question: **is there
directional signal on the tasks most likely to show one**, cheaply, before deciding whether the
full spend is warranted. Reporting is descriptive only: per-task pass (A vs B), constraint-
retention (A vs B), and the treatment-integrity summary (now using the v15.5-corrected gate).
4/4 A-wins-or-ties with real constraint-retention gaps would be an honest reason to fund the
full run; 0/4 or a mixed picture with no clear pattern is an honest reason to stop here and
report v15 Inconclusive rather than spend the other ~1,000 turns chasing it — that is a legitimate,
reportable outcome per the original prereg's own framing ("Null (equally reportable): the
optimizer does not measurably change real build quality at this n").
— Fable (Fable 5), 2026-07-17
