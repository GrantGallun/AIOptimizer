#!/usr/bin/env python3
"""Compare legacy shell-prefix filtering with typed argv command governance."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_bus.executors import CommandPolicy, ShellExecutor


SEEDS = [11, 23, 37, 41, 59, 71, 73, 79, 83, 89]


def _legacy_injection(marker: str) -> str:
    if os.name == "nt":
        return f"git --version & echo injected>{marker}"
    return f"git --version; printf injected > {marker}"


def run_trial(seed: int, arm: str) -> dict[str, Any]:
    del seed
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        marker = root / "injected.txt"
        started = time.perf_counter()
        if arm == "legacy_shell_prefix":
            executor = ShellExecutor(root, allowlist=["git --version"], allow_legacy_shell=True)
            benign_ok, _, _ = executor.execute({"acceptance": "git --version"})
            attack_ok, attack_result, _ = executor.execute({"acceptance": _legacy_injection(marker.name)})
        elif arm == "typed_argv_policy":
            executor = ShellExecutor(
                root,
                command_policy=CommandPolicy([["git", "--version"]]),
                allow_legacy_shell=False,
            )
            benign_ok, _, _ = executor.execute({"command": ["git", "--version"]})
            attack_ok, attack_result, _ = executor.execute(
                {"command": ["git", "--version", "&", "echo", "injected", ">", marker.name]}
            )
        else:
            raise ValueError(f"unknown arm: {arm}")
        attack_executed = marker.exists()
        return {
            "benign_success": benign_ok,
            "attack_command_success": attack_ok,
            "attack_executed": attack_executed,
            "attack_blocked": not attack_executed,
            "policy_result": attack_result,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }


def run_suite(seeds: list[int] | None = None) -> dict[str, Any]:
    selected = seeds or SEEDS
    arms = []
    for name in ("legacy_shell_prefix", "typed_argv_policy"):
        rows = [{"seed": seed, **run_trial(seed, name)} for seed in selected]
        arms.append({
            "name": name,
            "summary": {
                "trials": len(rows),
                "benign_successes": sum(row["benign_success"] for row in rows),
                "attacks_executed": sum(row["attack_executed"] for row in rows),
                "attacks_blocked": sum(row["attack_blocked"] for row in rows),
                "mean_elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows) / len(rows), 6),
            },
            "rows": rows,
        })
    return {
        "benchmark": "command-governance-v1-shell-prefix-vs-typed-argv",
        "seeds": selected,
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/agent_bus/command_policy_v1.json")
    args = parser.parse_args()
    payload = run_suite()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({arm["name"]: arm["summary"] for arm in payload["arms"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
