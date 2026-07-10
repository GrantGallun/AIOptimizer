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
- Still owed before concurrent execution: (1) the **board** needs the same lock (two cores calling
  `next` could double-dispatch a task → duplicate side effects); (2) **claim-before-edit** discipline
  on source files, not just cache keys; (3) **actually commit to git** — an unborn repo has no
  recovery, which is why Codex's file was unrecoverable. Cost governor is still relative tier units,
  not real token/dollar metering. `ShellExecutor` runs `acceptance` with `shell=True`: fine for
  human-authored tasks, but needs an allowlist before a model may populate that field.

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
