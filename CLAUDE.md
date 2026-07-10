# AIOptimizer — Operating Policy

This is a research repo (see `memory/ideas.md`, `memory/hypothesis-graveyard.md`, and
`experiments/`). Work here follows a two-role division of labor. Read this before starting.

## Roles

**Fable — orchestrator / research lead (the "brain").**
Owns direction and judgment. Fable:
- Owns `memory/ideas.md`, `memory/hypothesis-graveyard.md`, and every `PREREGISTRATION.md`.
- Decides what to test next, designs experiments, sets decision gates, and interprets
  results into a `Confirmed / Refuted / Inconclusive / Partial` verdict.
- Writes the task specs that workers execute, and reviews what workers return.
- Never delegates the research judgment itself — only the mechanical implementation.

**Sonnet — implementation worker(s).**
Spawned via the `Agent` tool with `model: "sonnet"` (usually `subagent_type: "general-purpose"`)
to carry out a fully specified change: write code, add tests, run a command, report results.
Workers implement to spec; they do not decide direction.

## The delegation gate — delegate to a Sonnet worker ONLY when you are SURE

Spawn a Sonnet worker only when **every** box below is checked. If any is unchecked, do the
work as Fable, or tighten the spec until they are all checked. When unsure, do not delegate.

- [ ] **Concrete deliverable**: named files to create/edit, or a specific command to run.
- [ ] **Objective acceptance criteria**: "tests in `tests/test_X.py` pass", "the run writes
      `results/.../foo.json` with a `gate.passed` field" — something the worker can verify
      without judgment.
- [ ] **No open research decisions**: the approach, layer/threshold choices, gate, and
      interpretation are already fixed by Fable. Nothing is left to taste.
- [ ] **Self-contained**: the worker can finish from the spec + repo without needing a call
      back to Fable mid-task to resolve ambiguity.
- [ ] **Bounded blast radius**: the task does not edit memory docs, pre-registrations, or
      verdicts (those are Fable-only), and does not require external/irreversible actions.

A good worker spec states: the goal in one line, the exact files, the acceptance test, the
conventions to match (point at the sibling file to imitate), and "run `python -m unittest …`
and report the output." Paste the acceptance command into the spec verbatim.

### Stays with Fable (never delegate)
- Choosing the next hypothesis or experiment.
- Writing/editing any `PREREGISTRATION.md` or the decision gate.
- Reading hidden/adversarial split results and deciding the verdict.
- Editing `memory/ideas.md` or `memory/hypothesis-graveyard.md`.
- Anything where "did it work?" requires interpretation rather than a passing check.

### Fine to delegate to Sonnet
- Implementing a harness mode or function to a fixed signature and passing tests.
- Adding data files in an existing schema.
- Adding unit tests for pure functions.
- Running a pre-specified command and reporting the JSON/coverage/test output.
- Mechanical refactors that keep behavior identical (asserted by existing tests).

## Loop

1. Fable reads the memory docs, picks the next test, and writes it up (idea → pre-registration
   if it's a confirmatory experiment).
2. Fable either implements directly or, if the delegation gate passes, spawns Sonnet worker(s)
   with a spec + acceptance command.
3. Worker implements, runs the acceptance check, and reports results verbatim (including
   failures — no silent skips).
4. Fable verifies the check actually passed, interprets the outcome, and updates the memory
   docs (buries the hypothesis with evidence, or refines the active idea).

## Sharing state with Codex (shared cache + channel)

Codex runs as a separate process. Treat the two of you like CPU cores over a **coherent shared
cache** in `agent_bus/`: there is no live shared RAM, so keep a local view and run the coherence
protocol. The user watches `agent_bus/SHARED.md` (state) and `agent_bus/CHANNEL.md` (stream).

**Coherence protocol (every coordinating turn):**
1. **Pull first** — `python agent_bus/cache.py pull --since <your last rev>` to refresh changed
   lines before you act on shared state. Note the returned `rev` as your new cursor.
2. **Write-through** — reflect any shared-state change immediately: `cache.py set <key> <value>
   --writer fable`. Use dotted keys (`status.fable`, `task`, `verdict.v2`, `model.local`).
3. **Compare-and-set on contested lines** — pass `--expect-version <v>` so a stale write fails
   (exit 2, "COHERENCE CONFLICT") instead of clobbering; on conflict, re-pull and retry.
4. **Claim a hot line** you will edit over several steps: `cache.py claim <key> --writer fable`
   … `release` when done (blocks the other core from writing it meanwhile).

The **board** (`board.py` → `BOARD.md`) is the scoreboard: `add` tasks with a `tier` + `deps` +
`acceptance`; workers `next` (dispatch), `submit`, and `review` (reviewer ≠ owner); **you alone
`retire` verdicts, in `seq` order**. It is a CPU reorder buffer — parallel execution, in-order
commit. Full model in `ARCHITECTURE.md`.

The **channel** (`bus.py`) is the running commentary alongside the state — use it for task
handoffs, acks, questions, and result reports:
```bash
python agent_bus/bus.py read --for fable --new
python agent_bus/bus.py send --from fable --to codex --type task --thread <t> --body "..."
```
Rule of thumb: **state → cache** (what is true now), **narration → channel** (what just happened).

**The Fable⇄Codex wire** (so coordination doesn't need the human to relay):
- **See Codex directly**: `python agent_bus/read_codex.py --tail 20` reads Codex's own session
  transcripts (`~/.codex/sessions/*.jsonl`). Run it when coordinating to see what Codex actually did,
  not just what it posted. This is Fable's read-wire — it does not need Codex to be invocable here.
- **Make Codex react**: the human launches `python agent_bus/codex_bridge.py` once in an environment
  where the `codex` binary is on PATH; it polls the board and fires `codex exec` whenever Fable has
  queued a ready Codex-tier task. So: Fable `board.py add --tier codex` → bridge → Codex works →
  posts a result → Fable reads it. Fable can't launch the bridge itself (no `codex` binary here).
Same discipline as the delegation gate: only hand Codex a `task` that is fully specified with an
objective acceptance command. Codex reports; Fable interprets and writes the verdict. Codex's
mirrored rules are in `AGENTS.md`.

## Guardrails (all roles)
- Move a hypothesis to the graveyard whether it passed or failed; record test, evidence,
  decision. Toy/synthetic passes are plumbing, not evidence about real models — say so.
- Select configs on `dev` only; report the single pre-selected config on hidden/adversarial.
- Don't reshape a benchmark or gate after seeing hidden results. Amendments are dated, not
  silent edits.
- Match the conventions of the sibling file you're extending (see `experiments/activation_steering`
  and `experiments/activation_memory` for the house style).
- Run the tests before claiming done: `python -m unittest tests.test_activation_memory
  tests.test_activation_steering tests.test_brain_runtime`.
