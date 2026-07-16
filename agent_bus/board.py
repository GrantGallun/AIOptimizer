#!/usr/bin/env python3
"""board — a scoreboard + reorder-buffer task dispatcher for the agent multicore.

The coordination model borrows a CPU's out-of-order core instead of inventing new
orchestration folklore:

  * **Scoreboard**: a task becomes dispatchable when its dependencies are satisfied, so
    a younger ready task can issue while an older blocked one waits (out-of-order issue).
  * **Heterogeneous dispatch**: each task carries a `tier` (which class of core may run
    it) — opus/fable, codex, sonnet, qwen — so work routes to the cheapest adequate core.
  * **Dual-modular redundancy**: `review` requires reviewer != owner (cross-check by a
    different core, ideally a different model family) before a task is VERIFIED.
  * **Reorder buffer / in-order retirement**: tasks execute out of order, but `retire`
    commits them in submission (`seq`) order — the research record (verdicts) stays
    consistent even though execution was parallel. Retirement is Fable-only.
  * **Watchdog**: a claimed-but-stale task is reclaimed so a crashed core can't deadlock.

State region for tasks (peer to `cache.py` = data memory, `bus.py` = interconnect).
`BOARD.md` renders the live scoreboard. Dependency-free (stdlib).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_bus.bus import _now
from agent_bus.cache import _cross_process_lock

OPS = ("spec", "impl", "test", "run", "review", "verdict", "integrate")
TIERS = ("fable", "codex", "sonnet", "qwen")
# Lifecycle: queued -> ready -> dispatched -> executed -> verified -> retired
#            (+ failed / squashed as terminal off-ramps)
ACTIVE = ("queued", "ready", "dispatched", "executed", "verified")
DONE_DEP = ("verified", "retired")  # a dependency counts as satisfied at these states


class BoardConflict(Exception):
    """A hazard: unmet dependency, self-review, out-of-order retirement, or stale ownership."""


class Board:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.store = self.root / "board.json"
        self.md = self.root / "BOARD.md"
        self.lockfile = self.root / ".board.lock"

    def _lock(self):
        return _cross_process_lock(self.lockfile)

    def _load(self) -> dict[str, Any]:
        if self.store.exists():
            return json.loads(self.store.read_text(encoding="utf-8"))
        return {"seq": 0, "tasks": {}}

    def _save(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._refresh_ready(state)
        tmp = self.store.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.store)
        self.render(state)

    def _refresh_ready(self, state: dict[str, Any]) -> None:
        tasks = state["tasks"]
        for task in tasks.values():
            if task["state"] == "queued":
                deps = task["deps"]
                if all(tasks.get(d, {}).get("state") in DONE_DEP for d in deps):
                    task["state"] = "ready"

    # -- issue ---------------------------------------------------------------
    def add(self, *, op: str, title: str, tier: str, spec: str = "", acceptance: str = "", command: list[str] | None = None, deps: list[str] | None = None, writes: list[str] | None = None, speculative: bool = False, branch: str | None = None) -> dict[str, Any]:
        if command is not None and (not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command)):
            raise ValueError("command must be a non-empty list of non-empty strings")
        with self._lock():
            state = self._load()
            state["seq"] += 1
            task_id = f"t{state['seq']:04d}"
            task = {
                "id": task_id,
                "seq": state["seq"],
                "op": op,
                "tier": tier,
                "title": title,
                "spec": spec,
                "acceptance": acceptance,
                "command": list(command) if command is not None else None,
                "deps": deps or [],
                "writes": sorted(set(writes or [])),
                "state": "queued",
                "owner": None,
                "reviewer": None,
                "result": None,
                "attempt": 0,
                "retry_feedback": "",
                "attempt_history": [],
                "speculative": speculative,
                "branch": branch,
                "ts": _now(),
                "lease_expires_at": None,
            }
            state["tasks"][task_id] = task
            self._save(state)
            return task

    def all(self) -> list[dict[str, Any]]:
        return sorted(self._load()["tasks"].values(), key=lambda t: t["seq"])

    def speculate(self, task_id: str, *, branch: str) -> dict[str, Any]:
        """Branch prediction: let a blocked task run ahead speculatively on a predicted branch."""
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            task["state"] = "ready"
            task["speculative"] = True
            task["branch"] = branch
            self._save(state)
            return task

    def commit(self, task_id: str) -> dict[str, Any]:
        """Prediction confirmed: clear the speculative flag so the task can eventually retire."""
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            task["speculative"] = False
            self._save(state)
            return task

    # -- scoreboard dispatch (out-of-order issue) ----------------------------
    def dispatch(self, *, tier: str, worker: str, lease_seconds: float = 900.0) -> dict[str, Any] | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        with self._lock():
            state = self._load()
            ready = [
                t for t in state["tasks"].values()
                if t["state"] == "ready" and t["tier"] == tier and t["owner"] is None
            ]
            if not ready:
                return None
            task = min(ready, key=lambda t: t["seq"])  # oldest ready first; blocked elders skipped
            task["state"] = "dispatched"
            task["owner"] = worker
            task["ts"] = _now()
            task["lease_expires_at"] = time.time() + lease_seconds
            self._save(state)
            return task

    def submit(self, task_id: str, *, worker: str, result: str) -> dict[str, Any]:
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            if task["owner"] != worker:
                raise BoardConflict(f"{task_id} is owned by {task['owner']}, not {worker}.")
            task["state"] = "executed"
            task["result"] = result
            task["ts"] = _now()
            task["lease_expires_at"] = None
            self._save(state)
            return task

    def defer(self, task_id: str, *, worker: str, reason: str = "") -> dict[str, Any]:
        """Return a dispatched task to ready when an external resource is unavailable."""
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            if task["state"] != "dispatched" or task["owner"] != worker:
                raise BoardConflict(f"{task_id} cannot be deferred by {worker}; owner/state changed.")
            task["state"] = "ready"
            task["owner"] = None
            task["result"] = f"deferred: {reason}"
            task["ts"] = _now()
            task["lease_expires_at"] = None
            self._save(state)
            return task

    # -- dual-modular redundancy (cross-check) -------------------------------
    def review(self, task_id: str, *, reviewer: str, ok: bool, note: str = "") -> dict[str, Any]:
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            if reviewer == task["owner"]:
                raise BoardConflict(f"{task_id} cannot be reviewed by its producer ({reviewer}); need a different core.")
            if task["state"] != "executed":
                raise BoardConflict(f"{task_id} is {task['state']}, not executed; nothing to review.")
            task["reviewer"] = reviewer
            task["state"] = "verified" if ok else "failed"
            task["result"] = f"{task['result']} | review({reviewer}): {'OK' if ok else 'REJECT'} {note}".strip()
            task["ts"] = _now()
            self._save(state)
            return task

    def retry(
        self,
        task_id: str,
        *,
        by: str,
        feedback: str = "",
    ) -> dict[str, Any]:
        """Return a rejected task to ready while preserving attempt evidence."""
        if not isinstance(by, str) or not by.strip():
            raise ValueError("retry actor must be non-empty")
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            if task["state"] != "failed":
                raise BoardConflict(
                    f"{task_id} is {task['state']}, not failed; only rejected work can retry."
                )
            history = list(task.get("attempt_history", []))
            history.append({
                "attempt": int(task.get("attempt", 0)),
                "owner": task.get("owner"),
                "reviewer": task.get("reviewer"),
                "result": task.get("result"),
                "ts": task.get("ts"),
            })
            task["attempt_history"] = history
            task["attempt"] = int(task.get("attempt", 0)) + 1
            task["retry_feedback"] = feedback or str(task.get("result") or "")
            task["state"] = "ready"
            task["owner"] = None
            task["reviewer"] = None
            task["result"] = f"retry {task['attempt']} scheduled by {by}"
            task["ts"] = _now()
            task["lease_expires_at"] = None
            self._save(state)
            return task

    # -- reorder buffer (in-order retirement, Fable only) --------------------
    def retire(self, task_id: str, *, by: str) -> dict[str, Any]:
        if by != "fable":
            raise BoardConflict("retirement commits the research record; Fable only.")
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            if task.get("speculative"):
                raise BoardConflict(f"{task_id} is still speculative; commit it once its branch resolves.")
            if task["state"] != "verified":
                raise BoardConflict(f"{task_id} is {task['state']}, not verified; cannot retire.")
            # In-order commit: any earlier task not yet on a terminal off-ramp blocks retirement.
            earlier = [t for t in state["tasks"].values() if t["seq"] < task["seq"] and t["state"] not in ("retired", "squashed", "failed")]
            if earlier:
                oldest = min(earlier, key=lambda t: t["seq"])
                raise BoardConflict(
                    f"in-order retirement: {oldest['id']} (seq {oldest['seq']}) is still {oldest['state']}; retire it first."
                )
            task["state"] = "retired"
            task["ts"] = _now()
            self._save(state)
            return task

    def park(self, task_id: str, *, reason: str = "") -> dict[str, Any]:
        """Set aside a task that needs the human/Fable (e.g. a verdict). Not redispatched."""
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            task["state"] = "blocked"
            task["result"] = f"awaiting: {reason}"
            task["ts"] = _now()
            self._save(state)
            return task

    def squash(self, task_id: str, *, reason: str = "") -> dict[str, Any]:
        with self._lock():
            state = self._load()
            task = self._require(state, task_id)
            task["state"] = "squashed"
            task["result"] = f"squashed: {reason}"
            task["ts"] = _now()
            self._save(state)
            return task

    # -- watchdog ------------------------------------------------------------
    def watchdog(self, *, now: float | None = None) -> list[str]:
        """Reclaim only dispatch leases that have expired; return reclaimed task ids."""
        with self._lock():
            state = self._load()
            current = time.time() if now is None else now
            reclaimed = []
            for task in state["tasks"].values():
                expires = task.get("lease_expires_at")
                if task["state"] == "dispatched" and expires is not None and expires <= current:
                    task["state"] = "ready"
                    task["owner"] = None
                    task["lease_expires_at"] = None
                    reclaimed.append(task["id"])
            if reclaimed:
                self._save(state)
            return reclaimed

    def get(self, task_id: str) -> dict[str, Any] | None:
        return self._load()["tasks"].get(task_id)

    def _require(self, state: dict[str, Any], task_id: str) -> dict[str, Any]:
        if task_id not in state["tasks"]:
            raise BoardConflict(f"no such task {task_id}.")
        return state["tasks"][task_id]

    # -- render --------------------------------------------------------------
    def render(self, state: dict[str, Any] | None = None) -> str:
        state = state if state is not None else self._load()
        tasks = sorted(state["tasks"].values(), key=lambda t: t["seq"])
        lines = [
            "# Scoreboard — agent multicore task board",
            "",
            f"_Out-of-order issue, in-order retirement. {len(tasks)} tasks · updated {_now()}._",
            "",
            "| id | seq | op | tier | state | owner | reviewer | deps | title |",
            "| --- | --: | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for t in tasks:
            deps = ",".join(t["deps"]) or "—"
            lines.append(
                f"| `{t['id']}` | {t['seq']} | {t['op']} | {t['tier']} | **{t['state']}** | "
                f"{t['owner'] or '—'} | {t['reviewer'] or '—'} | {deps} | {t['title'][:48]} |"
            )
        text = "\n".join(lines) + "\n"
        self.md.write_text(text, encoding="utf-8")
        return text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=str(Path(__file__).resolve().parent))
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Issue a task (Fable).")
    a.add_argument("--op", required=True, choices=OPS)
    a.add_argument("--title", required=True)
    a.add_argument("--tier", required=True, choices=TIERS)
    a.add_argument("--spec", default="")
    a.add_argument("--acceptance", default="")
    a.add_argument("--command-json", default="", help="Typed argv as a JSON string array.")
    a.add_argument("--deps", default="", help="Comma-separated task ids this depends on.")
    a.add_argument("--writes", default="", help="Comma-separated workspace paths this task may edit.")
    a.set_defaults(func=lambda b, ns: print(f"issued {b.add(op=ns.op, title=ns.title, tier=ns.tier, spec=ns.spec, acceptance=ns.acceptance, command=json.loads(ns.command_json) if ns.command_json else None, deps=[d for d in ns.deps.split(',') if d], writes=[p for p in ns.writes.split(',') if p])['id']}"))

    d = sub.add_parser("next", help="Dispatch the oldest ready task for a tier (a worker claims it).")
    d.add_argument("--tier", required=True, choices=TIERS)
    d.add_argument("--worker", required=True)
    d.set_defaults(func=lambda b, ns: print(json.dumps(b.dispatch(tier=ns.tier, worker=ns.worker), indent=2, ensure_ascii=False)))

    s = sub.add_parser("submit", help="Attach a result (owner).")
    s.add_argument("id"); s.add_argument("--worker", required=True); s.add_argument("--result", required=True)
    s.set_defaults(func=lambda b, ns: print(f"submitted {b.submit(ns.id, worker=ns.worker, result=ns.result)['state']}"))

    r = sub.add_parser("review", help="Cross-check (reviewer must differ from owner).")
    r.add_argument("id"); r.add_argument("--reviewer", required=True)
    r.add_argument("--ok", action="store_true"); r.add_argument("--note", default="")
    r.set_defaults(func=lambda b, ns: print(f"reviewed -> {b.review(ns.id, reviewer=ns.reviewer, ok=ns.ok, note=ns.note)['state']}"))

    rr = sub.add_parser("retry", help="Return independently rejected work to ready.")
    rr.add_argument("id"); rr.add_argument("--by", required=True); rr.add_argument("--feedback", default="")
    rr.set_defaults(func=lambda b, ns: print(f"retry -> {b.retry(ns.id, by=ns.by, feedback=ns.feedback)['state']}"))

    rt = sub.add_parser("retire", help="Commit in-order (Fable only).")
    rt.add_argument("id"); rt.add_argument("--by", required=True)
    rt.set_defaults(func=lambda b, ns: print(f"retired {b.retire(ns.id, by=ns.by)['id']}"))

    sq = sub.add_parser("squash", help="Cancel a task (e.g. speculation mispredict).")
    sq.add_argument("id"); sq.add_argument("--reason", default="")
    sq.set_defaults(func=lambda b, ns: print(f"squashed {b.squash(ns.id, reason=ns.reason)['id']}"))

    w = sub.add_parser("watchdog", help="Reclaim stale dispatched tasks.")
    w.set_defaults(func=lambda b, ns: print("reclaimed: " + (", ".join(b.watchdog()) or "none")))

    v = sub.add_parser("view", help="Regenerate BOARD.md.")
    v.set_defaults(func=lambda b, ns: (b.render(), print(f"rendered {b.md}")))

    return p


def main() -> None:
    ns = build_parser().parse_args()
    board = Board(Path(ns.root))
    try:
        ns.func(board, ns)
    except BoardConflict as exc:
        print(f"HAZARD: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
