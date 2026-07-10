# AIOptimizer — Project Plan (shared: Fable + Codex)

This is the **shared coordination surface** for the two agents working this repo:
- **Fable** (Claude, currently Opus 4.8) — research lead / orchestrator. Operating rules in `CLAUDE.md`.
- **Codex** — implementation + experiment execution. (If Codex uses an `AGENTS.md`, mirror the roles below there.)

Source of truth for *what is true* is still `memory/ideas.md` (open) and `memory/hypothesis-graveyard.md`
(tested). This file is for *what we are doing and who does it* — the roadmap and the handoff protocol.
Keep hypotheses in the memory docs; keep plans and coordination here.

**Signing convention (both agents):** every substantive edit to this file, the memory docs, or a
`PREREGISTRATION.md` ends with a dated signature line, e.g. `— Fable (Claude Opus 4.8), 2026-07-09`
or `— Codex, 2026-07-09`. Add a line to the Changelog at the bottom. This is how we attribute changes
without stepping on each other.

---

## 1. North star

Make **ordinary and local AI workers more reliable** by giving them a governed external memory and,
separately, by learning **where activation-space memory beats in-context memory**. The reliability we
care about is concrete and measurable: fewer stale facts, no privacy leaks, correct provenance, and —
the current open front — task accuracy that survives contact with a real model, not just a scorer.

## 2. Current state (2026-07-09)

| Thread | Where | Status | Latest evidence |
|---|---|---|---|
| Activation steering (behavior vectors) | `experiments/activation_steering` | Positive, preliminary | Orthogonalized vectors passed the anti-overfit gate on SmolLM2-135M (HYP-07) |
| Activation memory (memory-as-steering) | `experiments/activation_memory` | Registered, unrun | Toy gate passes by construction; **HF run is the real test** (HYP-AM-01) |
| Governed shared memory | `experiments/brain_runtime` | Structural win, end-to-end **refuted** | Structural 20/20 (HYP-09); Qwen3-8B **11/20 vs append-only 14/20** (HYP-10) |
| Local worker lane | `experiments/local_worker` | Working | Qwen3-8B via Ollama, ~3.3s/20-task suite, token+latency logged |

## 3. The live question (post-Qwen3)

Governed memory's **safety wins transferred** to the real model (0 leaks, 0 stale errors) but its
**accuracy win did not** (11/20 vs 14/20). Fable's diagnosis of the local run:

- The entire deficit is the **contradiction** case (governed 1/5 vs append-only 4/5). The runtime
  resolves the contradiction correctly; the **rendered prompt** then leads the small model to the
  wrong value → this is a *presentation* problem, not a memory-correctness problem.
- **private-scope** may be an ill-posed case: the legitimate answer appears reachable only behind the
  scoped-out private note, so governed is penalized for correctly withholding. **Verify before v2.**

So the goal is not "beat append-only on raw success" — it is: **keep the safety wins while closing the
presentation gap, and fix any unwinnable cases.**

## 4. Roadmap

### Now (this iteration)
- [x] **Fable** — QA'd the `private-scope` case: **not ill-posed.** The value is reachable and
      in-scope; the v1 failure is a source-label rendering bug (model echoed `public-test`). Folded
      into the v2 prereg rather than a case edit.
- [x] **Fable** — wrote `experiments/brain_runtime/PREREGISTRATION_v2.md`: retrieval-presentation
      ablation (C0 v1-repro / C1 resolved-only / C2 value-forward), splits, safety invariant, and the
      gate (governed ≥ append-only at 0 leaks / 0 stale on hidden seeds).
- [ ] **Sonnet worker (delegated by Fable)** — implement `presentation_ablation.py` + unit tests to
      the v2 prereg spec. Acceptance: `python -m unittest tests.test_presentation_ablation
      tests.test_brain_runtime` passes; no edits to v1 files.
- [x] **Fable** — ran the v2 ablation on Qwen3-8B. Verdict HYP-20260709-11: single-fact rendering
      fixes the contradiction bottleneck (C0 1/5 → C1 4/5 → C2 5/5; C2 total 15/20 > append-only 14/20);
      gate did not pass due to an invalid privacy probe (harness names the secret in the prompt). Two
      measurement flaws found, amended to v2.1 in the prereg.
- [ ] **Fable → Sonnet (next)** — implement v2.1: privacy-probe fix (secret only in an out-of-scope
      note, never named in the prompt) + de-confound `private-scope`, keep C2 rendering, re-run qwen3:8b.
      New module + result version; do not edit v2 files.
- [ ] **Fable** — run the `activation_memory` HF confirmatory pass (sweep dev → confirm hidden) once a
      model is available locally; record the HYP-AM-01 verdict.

### Next
- [x] **Orchestration**: real executors built (`agent_bus/executors.py`) — `ShellExecutor` (runs a
      task's acceptance command; drives test/run + Qwen-backed evals), `CodexExecutor` (`codex exec`),
      `RoutingExecutor` (verdict/fable → `NeedsHuman` human interrupt). Verified a LIVE non-sim loop:
      a real `python -m unittest` task executed + verified; Codex/verdict tasks parked for the human.
- [ ] **Live Codex loop (user's environment only)**: the `codex` binary isn't reachable from Fable's
      shells, so Fable cannot drive Codex. Run the headless loop where Codex is authed (`~/.codex`) —
      see the runbook below / `ARCHITECTURE.md`. Keep sandbox=workspace-write, a branch, and a budget.
- [ ] Rerun the v2 ablation on Qwen3-8B (Codex executes the pre-registered spec).
- [ ] If v2 closes the contradiction gap: replicate on a second local model (e.g. a 14B Q4) before any
      frontier call.
- [ ] Activation-memory: if HYP-AM-01 shows a crossover on 135M, replicate on a 1–3B instruct model.

### Later
- [ ] One **external** task/fact set neither agent authored (novelty claims need it).
- [ ] Fuse the threads: a memory whose *type* routes it to the better channel (governed text vs
      activation injection) — the payoff of activation_memory feeding brain_runtime.

## 5. Compute budget (project constraint)

- Hardware: RTX 5080, **16 GB VRAM** — the binding limit. Run Q4 ≤14B for GPU speed (8B ≈ 5–6 GB,
  14B ≈ 9–10 GB). 32B Q4 (~20 GB) spills; avoid.
- Disk: **budget ~50 GB free** (30 GB floor): Ollama + one 8B + one 14B + a fast 3–4B + small HF
  checkpoints (135M–3B fp16 are ≤6 GB each).
- Policy: **Qwen3-8B is the local regression worker.** Use it for every cheap repeatable run; reserve
  frontier/hosted models for the one high-value comparison after a local result holds.

## 6. Coordination protocol (how Fable and Codex avoid collisions)

- **Shared memory: coherent cache + channel.** Fable and Codex operate like two cores over a
  coherent shared cache in `agent_bus/`. **State → `cache.py`** (`pull` to refresh, `set` to
  write-through, `--expect-version` for compare-and-set, `claim`/`release` for line ownership);
  **narration → `bus.py`** (task handoffs, acks, questions, results). The user watches
  `agent_bus/SHARED.md` (state dashboard) and `agent_bus/CHANNEL.md` (message stream), both of
  which auto-reload in the IDE. Coherence protocol is in `CLAUDE.md` / `AGENTS.md`.
- **Ownership by lane, not by file.** Fable owns research judgment: `memory/*`, every
  `PREREGISTRATION*.md`, decision gates, and verdicts. Codex owns mechanical execution: harness code,
  data in an existing schema, running commands, reporting results verbatim. See `CLAUDE.md` for the
  full delegation gate.
- **Results are versioned, never overwritten.** `foo_v1.json` is frozen once written; a change makes
  `foo_v2.json`. Same for benchmark generators. Never retro-edit a suite after seeing its results.
- **One writer per file per session.** If you must touch the other agent's lane, leave a signed note in
  the Changelog saying what and why, so the other agent can reconcile.
- **Handoff format.** When Fable hands Codex a task, it names: the goal (one line), exact files, the
  acceptance command (verbatim), the sibling file to imitate, and "report output including failures."
  When Codex returns, it pastes the actual command output — no silent skips.
- **Verdicts are Fable's.** Codex reports numbers; Fable interprets them into
  Confirmed/Refuted/Inconclusive/Partial and updates the graveyard.

## 7. Open decisions for the user

- Approve budget for a second local model (14B Q4, ~10 GB) for the "Next" replication step?
- ~~Should Codex get a mirrored `AGENTS.md`?~~ Done (2026-07-09): `AGENTS.md` created with the bus protocol.

## Changelog

- 2026-07-09 — Created this plan; diagnosed the Qwen3-8B governed-memory refutation (loss isolated to
  the contradiction case; flagged private-scope as possibly ill-posed); set the Now/Next/Later roadmap
  and the Fable/Codex coordination protocol. — Fable (Claude Opus 4.8)
- 2026-07-09 — Confirmed Ollama 0.31.2 + qwen3:8b installed (added a PATH/VRAM note to the local_worker
  README); corrected the private-scope diagnosis (source-label rendering bug, not ill-posed); wrote
  PREREGISTRATION_v2.md; delegated the v2 harness implementation to a Sonnet worker. — Fable (Claude Opus 4.8)
- 2026-07-09 — Ran the v2 ablation on qwen3:8b; recorded HYP-20260709-11 (mechanism Confirmed — single-fact
  rendering fixes the contradiction bottleneck; combined gate Refuted on a harness-artifact leak). Amended the
  prereg to v2.1 (privacy-probe fix + de-confound private-scope). — Fable (Claude Opus 4.8)
- 2026-07-09 — Built the agent bus (`agent_bus/bus.py`, tests, `CHANNEL.md`) for bidirectional Fable⇄Codex
  messaging; created `AGENTS.md`; posted the v2.1 task to Codex (m0001). — Fable (Claude Opus 4.8)
- 2026-07-10 — Acted on Codex's architecture review. (1) Reconciled the two divergent v2.1s: a
  worker had overwritten Codex's better 5-case privacy/value split with a weaker 4-case version;
  reconstructed the split as canonical **v2.2** (design credit Codex) — value_forward **25/25** >
  append_only 21/25 on qwen3:8b, reproducing Codex's number (HYP-20260710-13). (2) Corrected the
  record: HYP-12 marked superseded/refined. (3) Hardened the state store: `cache.py` now takes a
  cross-process lock around read-modify-write (concurrency test added). Logged the open coordination
  gaps (board lock, claim-before-edit, git commits, real cost metering, shell allowlist) in
  ARCHITECTURE.md. — Fable (Claude Opus 4.8)
- 2026-07-09 — Ran the first **real loop iteration** end-to-end through the agent bus: Fable specced
  v2.1, a Sonnet worker built it (`presentation_ablation_v21.py`), Fable cross-checked, qwen3:8b ran
  the ablation, and the verdict retired in order. Result HYP-20260709-12: **value_forward 20/20, 0
  leaks, 0 stale, beating append_only 17/20** — "presentation was the v1 bottleneck" Confirmed on
  qwen3:8b (caveated: 1 model, 5 seeds, author-corrected harness → 14B replication next). — Fable (Claude Opus 4.8)
- 2026-07-09 — Built **real executors** (`agent_bus/executors.py`, tests): `ShellExecutor` (runs
  acceptance commands, model-free), `CodexExecutor` (`codex exec`), `RoutingExecutor` with a
  `NeedsHuman` interrupt for verdicts; added `board.park`. Verified a live non-sim loop (real
  `unittest` task executed + cross-checked; Codex/verdict tasks parked). Noted the boundary: the
  `codex` binary isn't reachable from Fable's shells, so the live Codex loop runs in the user's
  environment. — Fable (Claude Opus 4.8)
- 2026-07-09 — Built the **scheduler/driver** (`agent_bus/scheduler.py`, tests): heterogeneous
  dispatch, **speculative execution + branch prediction** (commit on hit, squash on mispredict), a
  **cost governor** (pre-admission budget control + maskable interrupt), and a watchdog. Ran a dry
  end-to-end loop exercising all mechanisms via a `SimExecutor` (no model calls); real executors
  (Sonnet sub-agent / `codex exec` / local Qwen) drop in behind the same interface next. — Fable (Claude Opus 4.8)
- 2026-07-09 — Added the **scoreboard + reorder-buffer task board** (`agent_bus/board.py`, tests,
  `BOARD.md`): out-of-order dependency-gated dispatch, heterogeneous tiers, cross-check requires a
  different core, in-order Fable-only retirement, watchdog. Wrote `ARCHITECTURE.md` mapping the whole
  agent system onto a heterogeneous OOO multicore (the "port the microarchitecture, don't reinvent
  it" design). — Fable (Claude Opus 4.8)
- 2026-07-09 — Upgraded coordination from a message log to a **coherent shared cache**
  (`agent_bus/cache.py`, tests, `SHARED.md`): versioned write-through, compare-and-set conflict
  detection, and claim/release line ownership (the "two cores + shared cache" model). Wired the
  coherence protocol into `CLAUDE.md` and `AGENTS.md`; seeded the shared state. — Fable (Claude Opus 4.8)
