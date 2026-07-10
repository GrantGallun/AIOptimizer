#!/usr/bin/env python3
"""Frozen v1 benchmark for lost updates across two scheduler drivers.

The unleased arm omits write declarations: both executors read the same file version,
then write, deterministically losing worker A's update.  The leased arm declares the
same path on both tasks: driver B is deferred while A owns the path, then retries and
observes A's committed update before writing its own.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_bus.board import Board
from agent_bus.scheduler import Scheduler


SEEDS = [11, 23, 37, 41, 59, 71, 73, 79, 83, 89]


class _UnleasedAppendExecutor:
    def __init__(self, target: Path, barrier: threading.Barrier, a_written: threading.Event) -> None:
        self.target = target
        self.barrier = barrier
        self.a_written = a_written

    def execute(self, task: dict[str, Any]) -> tuple[bool, str, float]:
        started = time.perf_counter()
        previous = self.target.read_text(encoding="utf-8")
        self.barrier.wait(timeout=5)
        token = task["title"]
        if token == "B":
            self.a_written.wait(timeout=5)
        self.target.write_text(previous + token, encoding="utf-8")
        if token == "A":
            self.a_written.set()
        return True, f"wrote {token}", time.perf_counter() - started


class _LeasedAppendExecutor:
    def __init__(self, target: Path, *, entered: threading.Event | None = None, proceed: threading.Event | None = None) -> None:
        self.target = target
        self.entered = entered
        self.proceed = proceed
        self.calls = 0

    def execute(self, task: dict[str, Any]) -> tuple[bool, str, float]:
        started = time.perf_counter()
        self.calls += 1
        previous = self.target.read_text(encoding="utf-8")
        if self.entered is not None:
            self.entered.set()
        if self.proceed is not None:
            self.proceed.wait(timeout=5)
        token = task["title"]
        self.target.write_text(previous + token, encoding="utf-8")
        return True, f"wrote {token}", time.perf_counter() - started


def _issue_pair(root: Path, *, declared: bool) -> tuple[str, str]:
    writes = ["artifact.txt"] if declared else []
    board = Board(root)
    first = board.add(op="impl", title="A", tier="codex", writes=writes)
    second = board.add(op="impl", title="B", tier="codex", writes=writes)
    return first["id"], second["id"]


def run_unleased_trial(seed: int) -> dict[str, Any]:
    del seed  # fixed orchestration; retained in rows for reproducible suite identity
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "artifact.txt"
        target.write_text("base:", encoding="utf-8")
        task_ids = _issue_pair(root, declared=False)
        barrier = threading.Barrier(2)
        a_written = threading.Event()
        executor = _UnleasedAppendExecutor(target, barrier, a_written)
        schedulers = [
            Scheduler(root, executor=executor, budget=100.0, scheduler_id=f"unleased-{index}")
            for index in range(2)
        ]
        threads = [threading.Thread(target=scheduler.tick) for scheduler in schedulers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        content = target.read_text(encoding="utf-8")
        return {
            "task_ids": list(task_ids),
            "content": content,
            "integrity": content == "base:AB",
            "lost_updates": int(content != "base:AB"),
            "deferred": 0,
        }


def run_leased_trial(seed: int) -> dict[str, Any]:
    del seed
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "artifact.txt"
        target.write_text("base:", encoding="utf-8")
        task_ids = _issue_pair(root, declared=True)
        entered = threading.Event()
        proceed = threading.Event()
        executor_a = _LeasedAppendExecutor(target, entered=entered, proceed=proceed)
        executor_b = _LeasedAppendExecutor(target)
        scheduler_a = Scheduler(root, executor=executor_a, budget=100.0, scheduler_id="leased-a")
        scheduler_b = Scheduler(root, executor=executor_b, budget=100.0, scheduler_id="leased-b")

        thread = threading.Thread(target=scheduler_a.tick)
        thread.start()
        if not entered.wait(timeout=5):
            raise RuntimeError("driver A did not enter executor")
        conflict_events = scheduler_b.tick()
        proceed.set()
        thread.join(timeout=10)
        retry_events = scheduler_b.tick()
        content = target.read_text(encoding="utf-8")
        return {
            "task_ids": list(task_ids),
            "content": content,
            "integrity": content == "base:AB",
            "lost_updates": int(content != "base:AB"),
            "deferred": len(conflict_events["deferred"]) + len(retry_events["deferred"]),
            "executor_b_calls": executor_b.calls,
        }


def run_suite(seeds: list[int] | None = None) -> dict[str, Any]:
    selected = seeds or SEEDS
    started = time.perf_counter()
    arms = []
    for name, runner in (("unleased", run_unleased_trial), ("leased_write_sets", run_leased_trial)):
        rows = [{"seed": seed, **runner(seed)} for seed in selected]
        arms.append(
            {
                "name": name,
                "summary": {
                    "trials": len(rows),
                    "integrity_successes": sum(row["integrity"] for row in rows),
                    "lost_updates": sum(row["lost_updates"] for row in rows),
                    "deferred": sum(row["deferred"] for row in rows),
                },
                "rows": rows,
            }
        )
    return {
        "benchmark": "workspace-collision-v1-two-scheduler-drivers",
        "seeds": selected,
        "mechanism": "atomic declared write-set leases",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/agent_bus/workspace_collision_v1.json")
    args = parser.parse_args()
    payload = run_suite()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({arm["name"]: arm["summary"] for arm in payload["arms"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
