#!/usr/bin/env python3
"""Analyze accuracy by within-run operator occurrence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _occurrence_totals(rows: list[dict]) -> dict[int, tuple[int, int]]:
    seen: dict[Any, int] = defaultdict(int)
    correct_by_occurrence: dict[int, int] = defaultdict(int)
    total_by_occurrence: dict[int, int] = defaultdict(int)

    for row in rows:
        operator = row["operator"]
        seen[operator] += 1
        occurrence = seen[operator]
        total_by_occurrence[occurrence] += 1
        correct_by_occurrence[occurrence] += int(bool(row["correct"]))

    return {
        occurrence: (correct_by_occurrence[occurrence], total)
        for occurrence, total in sorted(total_by_occurrence.items())
    }


def occurrence_accuracy(rows: list[dict]) -> dict[int, float]:
    """Return accuracy grouped by each operator's occurrence index."""
    return {
        occurrence: correct / total
        for occurrence, (correct, total) in _occurrence_totals(rows).items()
    }


def _filter_arm(rows: list[dict], arm: str | None) -> list[dict]:
    if arm is None or not any("arm" in row for row in rows):
        return rows
    return [row for row in rows if row.get("arm") == arm]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--arm")
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    rows = _filter_arm(payload.get("rows", []), args.arm)
    for occurrence, (correct, total) in _occurrence_totals(rows).items():
        print(f"occurrence {occurrence}: acc={correct / total:.3f} (n={total})")


if __name__ == "__main__":
    main()
