#!/usr/bin/env python3
"""Compare legacy whole-tree retirement with declared write-set retirement."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_bus.scheduler import GitCommitter


SEEDS = [11, 23, 37, 41, 59, 71, 73, 79, 83, 89]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def _prepare(root: Path) -> None:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "benchmark@example.invalid")
    _git(root, "config", "user.name", "benchmark")
    for name in ("target.txt", "staged-other.txt", "unstaged-other.txt"):
        (root / name).write_text("base", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "seed")
    (root / "target.txt").write_text("task change", encoding="utf-8")
    (root / "staged-other.txt").write_text("other staged change", encoding="utf-8")
    (root / "unstaged-other.txt").write_text("other unstaged change", encoding="utf-8")
    _git(root, "add", "staged-other.txt")


def run_trial(seed: int, arm: str) -> dict[str, Any]:
    del seed
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _prepare(root)
        if arm == "legacy_add_all":
            _git(root, "add", "-A")
            _git(root, "commit", "-m", "legacy retirement")
        elif arm == "declared_write_set":
            revision = GitCommitter(root)({
                "id": "t0001",
                "op": "impl",
                "tier": "codex",
                "title": "scoped edit",
                "writes": ["target.txt"],
            })
            if not revision:
                raise RuntimeError("scoped committer did not produce a revision")
        else:
            raise ValueError(f"unknown arm: {arm}")

        head = {
            name: _git(root, "show", f"HEAD:{name}").stdout
            for name in ("target.txt", "staged-other.txt", "unstaged-other.txt")
        }
        staged = set(_git(root, "diff", "--cached", "--name-only").stdout.splitlines())
        unstaged = set(_git(root, "diff", "--name-only").stdout.splitlines())
        return {
            "target_committed": head["target.txt"] == "task change",
            "unrelated_committed": sum(
                head[name] != "base" for name in ("staged-other.txt", "unstaged-other.txt")
            ),
            "staged_other_preserved": "staged-other.txt" in staged,
            "unstaged_other_preserved": "unstaged-other.txt" in unstaged,
        }


def run_suite(seeds: list[int] | None = None) -> dict[str, Any]:
    selected = seeds or SEEDS
    started = time.perf_counter()
    arms = []
    for name in ("legacy_add_all", "declared_write_set"):
        rows = [{"seed": seed, **run_trial(seed, name)} for seed in selected]
        arms.append({
            "name": name,
            "summary": {
                "trials": len(rows),
                "target_commits": sum(row["target_committed"] for row in rows),
                "unrelated_files_committed": sum(row["unrelated_committed"] for row in rows),
                "staged_other_preserved": sum(row["staged_other_preserved"] for row in rows),
                "unstaged_other_preserved": sum(row["unstaged_other_preserved"] for row in rows),
            },
            "rows": rows,
        })
    return {
        "benchmark": "retirement-commit-scope-v1",
        "seeds": selected,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/agent_bus/commit_scope_v1.json")
    args = parser.parse_args()
    payload = run_suite()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({arm["name"]: arm["summary"] for arm in payload["arms"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
