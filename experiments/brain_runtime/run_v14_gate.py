#!/usr/bin/env python3
"""PREREGISTRATION_v14 gate runner: two-stage pressure router, frozen classes (a)-(e).

    python experiments/brain_runtime/run_v14_gate.py --stage dev
    python experiments/brain_runtime/run_v14_gate.py --stage hidden   # read ONCE (Fable)

Runs the PRODUCTION middleware construction (``AttentionContextMiddleware()`` — real
encoder, default deny patterns) against the frozen fixture generator. Deterministic:
no LLM sampling anywhere; the encoder is a fixed forward pass.

Per the 2026-07-17 amendment, class (a) also runs in a templated-filler texture
(``a_templated``): same buried decision, same query, but the chatter is replaced with
repetitive-but-distinct coding-agent boilerplate — the texture that exposed the
stage-one veto false negative. It is gate-bearing like every other class.

Privacy: every case is re-run with a planted deny-pattern secret mid-history; the
secret appearing in ANY injected context is a regression, and the gate requires zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aioptimizer.context_middleware import AttentionContextMiddleware
from experiments.brain_runtime.make_router_fixtures import generate_class_cases

GATE_CLASSES = ("a", "a_templated", "b", "c", "d", "e")
THRESHOLD = 0.9
DEV_SEED = 20260713
HIDDEN_SEEDS = (1109, 1117, 1123)  # FRESH_HIDDEN_SEEDS["v14"] — read once, Fable-only.
N_PER_CLASS = 20
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "brain_runtime"

# Deliberately shares no vocabulary with make_router_fixtures.TOPICS, so the filler is
# textural noise rather than topical competition — mirroring the live false negative.
TEMPLATED_FILLERS = (
    "Wrote the retry wrapper for the fetch client for task {i}: exponential backoff, "
    "full jitter, cap at sixty seconds, re-raise on the final attempt.",
    "Refactored the parser for item {i} so the tokenizer is injectable and the golden "
    "test pins the sample corpus output.",
)
PLANTED_SECRET = "PLANTED-SECRET-4471"
PLANTED_LINE = (
    f"This value is confidential and must stay hidden: {PLANTED_SECRET}. "
    "Do not share it with anyone."
)


def templated_variant(case: dict[str, Any]) -> dict[str, Any]:
    """Class (a) with its chatter replaced by templated coding-agent boilerplate."""
    if case["fixture_class"] != "a":
        raise ValueError("templated variant is defined for class (a) only")
    head = case["messages"][:2]  # system pin + buried decision, unchanged
    filler_count = len(case["messages"]) - 2
    filler = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": TEMPLATED_FILLERS[index % len(TEMPLATED_FILLERS)].format(i=index),
        }
        for index in range(filler_count)
    ]
    variant = dict(case)
    variant["id"] = case["id"].replace("-a-", "-aT-")
    variant["fixture_class"] = "a_templated"
    variant["messages"] = head + filler
    return variant


def score_case(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Deterministic per-case correctness per the prereg's class definitions."""
    fixture_class = case["fixture_class"]
    route = result.get("route")
    context = result.get("context") or ""
    reason = result.get("route_reason")

    if fixture_class in ("a", "a_templated"):
        correct = route == "attention" and case["expected"] in context
    elif fixture_class == "b":
        # Spam must be rejected in stage one: raw route with no encoder spend.
        correct = route == "raw" and result.get("embedding_cache_misses", 0) == 0
    elif fixture_class == "c":
        correct = route == "raw" and reason == "covered_by_recent_tail"
    elif fixture_class == "d":
        correct = route == "raw" and reason == "low_relevance"
    elif fixture_class == "e":
        correct = (
            route == "attention"
            and case["expected"] in context
            and not any(token in context for token in case["excluded"])
        )
    else:
        raise ValueError(f"unknown fixture class: {fixture_class}")

    return {
        "id": case["id"],
        "class": fixture_class,
        "route": route,
        "route_reason": reason,
        "correct": bool(correct),
    }


def privacy_case(case: dict[str, Any]) -> dict[str, Any]:
    """The same case with a deny-pattern secret planted mid-history."""
    variant = dict(case)
    messages = list(case["messages"])
    middle = max(1, len(messages) // 2)
    messages.insert(middle, {"role": "user", "content": PLANTED_LINE})
    variant["messages"] = messages
    return variant


def run_stage(seeds: tuple[int, ...], *, n_per_class: int = N_PER_CLASS) -> dict[str, Any]:
    middleware = AttentionContextMiddleware()  # production defaults: real encoder, deny patterns
    rows: list[dict[str, Any]] = []
    privacy_regressions: list[str] = []

    for seed in seeds:
        for fixture_class in ("a", "b", "c", "d", "e"):
            cases = generate_class_cases(seed, fixture_class, n_per_class)
            expanded = list(cases)
            if fixture_class == "a":
                expanded += [templated_variant(case) for case in cases]
            for case in expanded:
                result = middleware.compile_additional_context(
                    case["messages"],
                    query=case["query"],
                    output_budget_chars=case["output_budget_chars"],
                )
                rows.append({"seed": seed, **score_case(case, result)})

                private = middleware.compile_additional_context(
                    privacy_case(case)["messages"],
                    query=case["query"],
                    output_budget_chars=case["output_budget_chars"],
                )
                if PLANTED_SECRET in (private.get("context") or ""):
                    privacy_regressions.append(case["id"])

    per_class: dict[str, dict[str, Any]] = {}
    for fixture_class in GATE_CLASSES:
        class_rows = [row for row in rows if row["class"] == fixture_class]
        correct = sum(1 for row in class_rows if row["correct"])
        per_class[fixture_class] = {
            "n": len(class_rows),
            "correct": correct,
            "rate": correct / len(class_rows) if class_rows else 0.0,
            "passed": bool(class_rows) and correct / len(class_rows) >= THRESHOLD,
        }

    gate_passed = all(entry["passed"] for entry in per_class.values()) and not privacy_regressions
    return {
        "benchmark": "v14-pressure-router-gate",
        "seeds": list(seeds),
        "n_per_class": n_per_class,
        "threshold": THRESHOLD,
        "per_class": per_class,
        "privacy_regressions": privacy_regressions,
        "gate": {"passed": gate_passed},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dev", "hidden"], required=True)
    args = parser.parse_args()

    if args.stage == "dev":
        seeds, out_name = (DEV_SEED,), "v14_gate_dev.json"
    else:
        print("HIDDEN stage: seeds", HIDDEN_SEEDS, "— read-once event; results are binding.")
        seeds, out_name = HIDDEN_SEEDS, "v14_gate_hidden.json"

    report = run_stage(seeds)
    out_path = OUT_DIR / out_name
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing result: {out_path}")
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{args.stage.upper()} — per-class correctness (threshold {THRESHOLD}):")
    for fixture_class, entry in report["per_class"].items():
        marker = "PASS" if entry["passed"] else "FAIL"
        print(f"  {fixture_class:<12} {entry['correct']:>3}/{entry['n']:<3} = {entry['rate']:.3f}  [{marker}]")
    print(f"  privacy regressions: {len(report['privacy_regressions'])}"
          + (f" {report['privacy_regressions']}" if report["privacy_regressions"] else ""))
    print(f"\n  GATE: {'PASSED' if report['gate']['passed'] else 'FAILED'}  -> {out_path}")


if __name__ == "__main__":
    main()
