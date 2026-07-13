#!/usr/bin/env python3
"""Arm-blind objective scorer for the v15 dogfood A/B (PREREGISTRATION_v15).

Codex's runner drives each scripted task through arm A (plugin ON) and arm B (plugin OFF)
in the pristine paired workspaces, producing one artifact directory per (task, arm). This
scorer grades those directories with IDENTICAL deterministic checks — it never receives arm
identity while scoring (the runner names the two dirs; the checks are symmetric), so the
grade is arm-blind by construction.

Per task, the objective checks are:
  - file_presence: every expected relative path exists and is non-empty
  - requirement contracts: aioptimizer.requirements.evaluate_requirements over the combined
    artifact text (must_include / must_exclude), reusing the SAME engine the gateway ships
  - ast_checks: light Python AST assertions (defines/imports/forbids) on named files
A task PASSES iff files present AND all NON-constraint contracts pass AND all ast_checks pass.
CONSTRAINT contracts (tagged is_constraint) feed the separate constraint-retention metric —
the thing the injected context is supposed to preserve across a long prompt script.

    python experiments/brain_runtime/score_v15.py --tasks v15_build_tasks.json \
        --arm-a <dir> --arm-b <dir> --out results/brain_runtime/v15_scored.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aioptimizer.requirements import Requirement, evaluate_requirements
from experiments.brain_runtime.stats import wilson_interval


def _artifact_text(arm_dir: Path, task: Mapping[str, Any]) -> tuple[str, bool]:
    """Concatenate the task's expected files; also report whether all are present+non-empty."""
    parts, all_present = [], True
    for rel in task["expected_files"]:
        path = arm_dir / task["id"] / rel
        if path.exists() and path.read_text(encoding="utf-8", errors="replace").strip():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        else:
            all_present = False
    return "\n\n".join(parts), all_present


def _ast_check(text: str, check: Mapping[str, Any]) -> bool:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imports |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    kind = check["kind"]
    if kind == "defines":
        return check["name"] in names
    if kind == "imports":
        return check["name"] in imports
    if kind == "forbid_bare_except":
        return not any(isinstance(h, ast.ExceptHandler) and h.type is None for h in ast.walk(tree))
    raise ValueError(f"unknown ast check: {kind}")


def _to_requirements(rows: list[Mapping[str, Any]]) -> tuple[Requirement, ...]:
    return tuple(
        Requirement(
            id=r["id"],
            must_include=tuple(r.get("must_include", [])),
            must_exclude=tuple(r.get("must_exclude", [])),
            case_sensitive=bool(r.get("case_sensitive", False)),
        )
        for r in rows
    )


def score_arm(arm_dir: Path, tasks: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for task in tasks:
        text, files_ok = _artifact_text(arm_dir, task)
        contracts = task.get("requirements", [])
        constraint_rows = [c for c in contracts if c.get("is_constraint")]
        functional_rows = [c for c in contracts if not c.get("is_constraint")]
        func_eval = evaluate_requirements(text, _to_requirements(functional_rows))
        cons_eval = evaluate_requirements(text, _to_requirements(constraint_rows))
        ast_ok = all(_ast_check(text, c) for c in task.get("ast_checks", []))
        functional_ok = func_eval is None or func_eval["all_passed"]
        passed = files_ok and functional_ok and ast_ok
        rows.append({
            "task": task["id"],
            "files_present": files_ok,
            "functional_ok": functional_ok,
            "ast_ok": ast_ok,
            "passed": passed,
            "constraints_total": (cons_eval or {}).get("requirements", 0),
            "constraints_passed": (cons_eval or {}).get("passed", 0),
        })
    return {"rows": rows}


def summarize(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    by = {"A": a["rows"], "B": b["rows"]}
    out: dict[str, Any] = {"arms": {}}
    for arm, rows in by.items():
        n = len(rows)
        passed = sum(r["passed"] for r in rows)
        ct = sum(r["constraints_total"] for r in rows)
        cp = sum(r["constraints_passed"] for r in rows)
        lo, hi = wilson_interval(passed, n) if n else (0.0, 0.0)
        out["arms"][arm] = {
            "tasks": n, "passed": passed, "pass_rate": passed / n if n else 0.0,
            "pass_ci": [round(lo, 4), round(hi, 4)],
            "constraint_retention": cp / ct if ct else None,
        }
    ra, rb = out["arms"]["A"], out["arms"]["B"]
    out["pass_gap_A_minus_B"] = round(ra["pass_rate"] - rb["pass_rate"], 4)
    # paired per-task deltas (A better / B better / tie)
    pa = {r["task"]: r["passed"] for r in a["rows"]}
    pb = {r["task"]: r["passed"] for r in b["rows"]}
    out["paired"] = {
        "A_only": sorted(t for t in pa if pa[t] and not pb.get(t)),
        "B_only": sorted(t for t in pb if pb[t] and not pa.get(t)),
        "both": sorted(t for t in pa if pa[t] and pb.get(t)),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--arm-a", required=True, help="Artifact dir for one arm.")
    parser.add_argument("--arm-b", required=True, help="Artifact dir for the other arm.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    a = score_arm(Path(args.arm_a), tasks)
    b = score_arm(Path(args.arm_b), tasks)
    payload = {"tasks": len(tasks), "A": a, "B": b, "summary": summarize(a, b)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s = payload["summary"]
    print(f"A pass {s['arms']['A']['passed']}/{s['arms']['A']['tasks']} "
          f"({s['arms']['A']['pass_rate']:.2f}) CI {s['arms']['A']['pass_ci']}; "
          f"B pass {s['arms']['B']['passed']}/{s['arms']['B']['tasks']} "
          f"({s['arms']['B']['pass_rate']:.2f}); gap {s['pass_gap_A_minus_B']:+.2f}")


if __name__ == "__main__":
    main()
