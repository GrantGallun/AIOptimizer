# The recursive improvement loop — runbook

The loop improves the project — including *itself* (its own orchestration machinery) — under
hard rails. It is bounded self-improvement: every cycle is gated by tests, cross-checked by a
different model, verdict-committed only by Fable in order, cost-capped, and durably committed to
git so nothing is lost. Full architecture in `ARCHITECTURE.md`.

## One cycle

```
FETCH            DECODE               EXECUTE                     RETIRE
next ready task → Fable spec + gate → worker runs + cross-check → Fable commits (git) in order
```

Concretely: `scheduler.tick()` pulls ready tasks by tier, an `Executor` runs each, a *different*
core reviews it (`review`, reviewer ≠ owner), and each in-order `retire` fires `GitCommitter` →
a local `git commit`. Verdicts and Codex-less impl tasks **park** for the human (NeedsHuman).

## The rails (never disabled)

1. **Cost governor** — pre-admission budget (`--budget`); trips a maskable interrupt.
2. **Cross-check** — `review` requires reviewer ≠ owner, ideally a different model family.
3. **In-order retirement** — only Fable retires, in `seq` order; retirement = the git commit.
4. **Human interrupt** — verdicts (`op=verdict`/`tier=fable`) never auto-run; they park for you.
5. **Durability** — `--commit` commits locally per retired task (recoverable); **never pushes**.

## Cores are heterogeneous across *and within* vendors

The tiers aren't four fixed models — they're a *set* of cores, and each vendor can contribute
several. Codex takes `-m <model>`, so you can register multiple Codex cores at different models
(a cheap/fast GPT for simple `impl`, a stronger one for hard tasks) via `CodexExecutor`'s
`extra_args`:
```bash
CodexExecutor(repo, extra_args=["-m", "gpt-5"])          # a "codex-strong" core
CodexExecutor(repo, extra_args=["-m", "<fast-model>"])   # a "codex-fast" core
```
So the ladder is: Claude cores (Opus judgment / Sonnet throughput) + N Codex cores (by GPT model)
+ local (Qwen). This improves both cost-routing (send each task to the cheapest adequate core) and
cross-check quality (reviewer can be a genuinely different vendor *and* model than the producer).

## Levels — pick your autonomy

### Level 1 — single-driver backlog drain (runnable today)
Fable seeds the board; one driver drains it. Command tasks (`test`/`run`, incl. Qwen evals) run
locally; `impl` tasks go to Codex if enabled, else park for Fable/Sonnet fan-out.
```bash
# seed a task (Fable):
python agent_bus/board.py add --op impl --tier codex --title "..." --spec "..." --acceptance "python -m unittest tests.test_X"
# drain it, committing each retirement, budget-capped:
python agent_bus/run_loop.py --budget 10 --commit           # local shell/qwen cores only
python agent_bus/run_loop.py --budget 10 --commit --codex --codex-args "--full-auto"   # + Codex core (its env)
```
Stops when the board has no `ready` tasks, the budget trips, or only parked tasks remain.

### Level 2 — self-replenishing (semi-auto)
When the board drains, it needs judgment: Fable reads `memory/ideas.md` + the graveyard, proposes
the next batch of specced tasks, seeds them, and the loop continues. This is the honest recursive
form — it runs until it needs a decision, then asks. Claude can run this via the `/loop` skill
(re-enter each interval: drain → if empty, propose+seed → repeat).

### Level 3 — dual-driver full-auto (the ambition)
Two drivers over one board at once: your Codex-headless `run_loop.py --codex` **and** a Claude-side
`/loop` fanning out Sonnet workers. Cross-check spans model families automatically. **Prerequisite:**
the **board cross-process lock** (task 1 below) — without it two drivers can double-dispatch. Keep a
tight `--budget` and the verdict IRQ.

## Current backlog (seeded on the board — the loop's fuel)

The recursive part: the first three tasks harden the loop's own machinery.
1. **Board cross-process lock** — port `cache.py`'s `_cross_process_lock` to every mutating
   `board.py` method (prereq for Level 3). Acceptance: `tests.test_board` + a double-dispatch test.
2. **ShellExecutor allowlist** — guard `acceptance` so only allowlisted commands run before a model
   may populate that field. Acceptance: a test that a disallowed command is refused.
3. **Real cost metering** — charge actual tokens/duration from executor results, not static tier
   cost, so the governor is a real spend cap. Acceptance: a metering test.
4. **14B-Q4 replication of v2.2** (research) — rerun `presentation_ablation_v22.py` on a 14B Q4
   model. Needs the model pulled (human). Acceptance: a `_14b.json` result with the gate.
5. **Leak-eliciting privacy variant** (research) — a privacy case that actually induces leaks, so
   governed memory's structural privacy advantage can be *demonstrated*, not just asserted.

Verdicts on 4–5 are Fable's, in order, human-gated.

## Honest boundary

Neither Claude nor Codex is a daemon; each acts in turns. So the loop is driver-agnostic: a human,
a Claude `/loop`, or a Codex headless loop all drive the same board under the same rails. "Recursive
self-improvement" here means the system safely improves its own machinery and research — not
unbounded autonomy. The rails are the point.
