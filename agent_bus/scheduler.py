#!/usr/bin/env python3
"""scheduler — the OS scheduler / driver for the agent multicore.

Ties the pieces into a running loop, borrowing three more CPU mechanisms:

  * **Heterogeneous scheduling**: each tick, every core class (fable / codex / sonnet / qwen)
    with capacity pulls its next ready task off the scoreboard and runs it.
  * **Speculative execution + branch prediction**: when a gate task (test/run/verdict) is
    dispatched, the predictor guesses its outcome and lets dependent work run *ahead* on cheap
    E-cores. On a correct prediction the work is committed; on a mispredict it is squashed.
  * **Cost governor (maskable interrupt)**: every execution is charged; crossing the budget
    trips an interrupt that halts dispatch (verdicts/frontier spend already require the human).

Execution goes through an `Executor` so the loop is driver-agnostic. `SimExecutor` completes
tasks deterministically for dry runs (no model calls); real executors (a Sonnet sub-agent, a
`codex exec` call, a local Qwen run) drop in behind the same interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_bus.board import Board, BoardConflict
from agent_bus.cache import Cache
from agent_bus.executors import NeedsHuman

# Relative cost per core class (E-cores are ~free, the P-core is dear).
TIER_COST = {"fable": 1.0, "codex": 0.4, "sonnet": 0.1, "qwen": 0.01}
# Cross-check routing: reviewer is a *different* core class (ideally a different model family).
REVIEWER = {"sonnet": "codex", "codex": "fable", "qwen": "codex", "fable": "codex"}
GATE_OPS = ("test", "run", "verdict")


class Governor:
    """Cost accounting with a budget ceiling; trips a maskable interrupt when exceeded."""

    def __init__(self, cache: Cache, budget: float) -> None:
        self.cache = cache
        self.budget = budget
        self.spent = 0.0
        self._publish()

    def charge(self, cost: float) -> None:
        self.spent += cost
        self._publish()

    def tripped(self) -> bool:
        return self.spent >= self.budget

    def would_exceed(self, cost: float) -> bool:
        """Pre-admission control: refuse to start work that wouldn't fit the remaining budget."""
        return self.spent + cost > self.budget

    def _publish(self) -> None:
        # Surface the budget on the shared dashboard so the human can see the IRQ approach.
        self.cache.set("governor.budget", f"{self.budget:.2f}", writer="fable", scope="governor")
        self.cache.set("governor.spent", f"{self.spent:.2f}", writer="fable", scope="governor")


class SimExecutor:
    """Deterministic executor for dry runs. `fail` marks task ids whose gate should mispredict."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()

    def execute(self, task: dict[str, Any]) -> tuple[bool, str, float]:
        ok = task["id"] not in self.fail
        cost = TIER_COST.get(task["tier"], 0.5)
        verdict = "pass" if ok else "FAIL"
        return ok, f"[sim] {task['op']} '{task['title']}' -> {verdict}", cost


class BranchPredictor:
    """Predict a gate's outcome so dependents can run ahead. Default: predict pass (taken)."""

    def __init__(self, predict_pass: Callable[[dict[str, Any]], bool] | None = None) -> None:
        self._predict = predict_pass or (lambda task: True)

    def predict(self, gate: dict[str, Any]) -> bool:
        return self._predict(gate)


class Scheduler:
    def __init__(
        self,
        root: Path,
        *,
        cores: dict[str, int] | None = None,
        executor: Any | None = None,
        budget: float = 10.0,
        predictor: BranchPredictor | None = None,
    ) -> None:
        self.board = Board(root)
        self.cache = Cache(root)
        self.cores = cores or {"fable": 1, "codex": 1, "sonnet": 2, "qwen": 1}
        self.executor = executor or SimExecutor()
        self.governor = Governor(self.cache, budget)
        self.predictor = predictor or BranchPredictor()

    def _reviewer_for(self, task: dict[str, Any]) -> str:
        return f"{REVIEWER.get(task['tier'], 'codex')}-review"

    def _speculate_dependents(self, gate: dict[str, Any], events: dict[str, Any]) -> None:
        if not self.predictor.predict(gate):
            return
        branch = f"{gate['id']}:pass"
        for t in self.board.all():
            if gate["id"] in t["deps"] and t["state"] == "queued" and t["tier"] in ("sonnet", "qwen"):
                self.board.speculate(t["id"], branch=branch)
                events["speculated"].append(t["id"])

    def _resolve_speculation(self, gate: dict[str, Any], gate_ok: bool, events: dict[str, Any]) -> None:
        branch = f"{gate['id']}:pass"
        for t in self.board.all():
            if t.get("branch") != branch:
                continue
            if gate_ok:
                if t.get("speculative"):
                    self.board.commit(t["id"])  # prediction correct -> keep the work
                    events["committed"].append(t["id"])
            else:
                self.board.squash(t["id"], reason=f"mispredicted branch {branch}")
                events["squashed"].append(t["id"])

    def _retire_in_order(self, events: dict[str, Any]) -> None:
        for t in self.board.all():
            if t["state"] == "verified" and not t.get("speculative"):
                try:
                    self.board.retire(t["id"], by="fable")
                    events["retired"].append(t["id"])
                except BoardConflict:
                    break  # an earlier task hasn't retired yet; stop (in-order commit)

    def tick(self) -> dict[str, Any]:
        events = {k: [] for k in ("dispatched", "executed", "verified", "retired", "squashed", "speculated", "committed", "awaiting")}
        events["tripped"] = False
        self.board.watchdog()
        if self.governor.tripped():
            events["tripped"] = True
            return events
        for tier, capacity in self.cores.items():
            estimated = TIER_COST.get(tier, 0.5)
            for _ in range(capacity):
                if self.governor.would_exceed(estimated):
                    events["tripped"] = True  # power/cost budget: won't admit this core's work
                    break
                task = self.board.dispatch(tier=tier, worker=f"{tier}-core")
                if task is None:
                    break
                try:
                    ok, result, cost = self.executor.execute(task)
                except NeedsHuman:
                    self.board.park(task["id"], reason="needs Fable/human")
                    events["awaiting"].append(task["id"])  # human interrupt: judgment handed back
                    continue
                if task["op"] in GATE_OPS:
                    self._speculate_dependents(task, events)
                self.governor.charge(cost)
                events["dispatched"].append(task["id"])
                self.board.submit(task["id"], worker=task["owner"], result=result)
                events["executed"].append(task["id"])
                verified = self.board.review(task["id"], reviewer=self._reviewer_for(task), ok=ok)
                if verified["state"] == "verified":
                    events["verified"].append(task["id"])
                if task["op"] in GATE_OPS:
                    self._resolve_speculation(task, ok, events)
        self._retire_in_order(events)
        return events

    def run(self, *, max_ticks: int = 50) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for _ in range(max_ticks):
            ev = self.tick()
            history.append(ev)
            if ev["tripped"]:
                break
            progressed = any(ev[k] for k in ("dispatched", "retired", "squashed", "committed"))
            if not progressed:
                break  # board drained or stalled
        return history


def _summarize(history: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"ticks": len(history)}
    for key in ("dispatched", "executed", "verified", "retired", "squashed", "speculated", "committed", "awaiting"):
        agg[key] = sum(len(ev[key]) for ev in history)
    agg["tripped"] = any(ev["tripped"] for ev in history)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--max-ticks", type=int, default=50)
    parser.add_argument("--fail", default="", help="Comma-separated task ids whose gate should mispredict (dry-run demo).")
    args = parser.parse_args()
    executor = SimExecutor(fail={x for x in args.fail.split(",") if x})
    sched = Scheduler(Path(args.root), executor=executor, budget=args.budget)
    history = sched.run(max_ticks=args.max_ticks)
    print(json.dumps({"per_tick": history, "summary": _summarize(history)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
