#!/usr/bin/env python3
"""Deterministic prereg-v14 pressure-router fixtures.

Each seed produces cases for all five frozen classes.  The generator does not run
the router or inspect results, so fresh hidden seeds remain safe for one-shot use.

    python experiments/brain_runtime/make_router_fixtures.py --seed 20260713 \
        --n-cases-per-class 10 --out results/brain_runtime/router_fixtures_dev.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


FIXTURE_CLASSES = ("a", "b", "c", "d", "e")
TOPICS = (
    ("cache backend", "redis", "sqlite"),
    ("deployment region", "orion", "vega"),
    ("index format", "parquet", "arrow"),
    ("queue transport", "nats", "kafka"),
    ("release channel", "canary", "stable"),
    ("memory policy", "newest", "actr"),
)
DISTRACTOR_SUBJECTS = (
    "benchmark sampling", "documentation cleanup", "test isolation", "change notes",
    "dashboard colors", "meeting cadence", "dependency review", "schema naming",
    "worker telemetry", "build acceleration", "lint configuration", "artifact retention",
    "on-call rotation", "fixture coverage", "command parsing", "request tracing",
)
DISTRACTOR_ACTIONS = (
    "recorded a follow-up with concrete owners",
    "compared the alternatives and postponed the choice",
    "added a narrow test before changing defaults",
    "kept the existing behavior pending another sample",
    "captured measurements without drawing a verdict",
    "split the work into independently checkable pieces",
    "noted a compatibility risk for the next review",
    "verified the example against the current interface",
)


def _chatter(rng: random.Random, count: int, *, offset: int = 0) -> list[dict[str, str]]:
    messages = []
    for index in range(count):
        subject = rng.choice(DISTRACTOR_SUBJECTS)
        action = rng.choice(DISTRACTOR_ACTIONS)
        nonce = rng.randrange(10_000, 99_999)
        role = "user" if (index + offset) % 2 == 0 else "assistant"
        messages.append({
            "role": role,
            "content": (
                f"Worklog {nonce} on {subject}: the team {action}. "
                f"Checkpoint {index + 1} remains independent of product routing."
            ),
        })
    return messages


def generate_fixture(
    rng: random.Random,
    fixture_class: str,
    case_id: str,
) -> dict[str, Any]:
    """Generate one frozen class (a)-(e) case from ``rng``."""
    if fixture_class not in FIXTURE_CLASSES:
        raise ValueError(f"unknown fixture class: {fixture_class}")
    topic, old_value, current_value = rng.choice(TOPICS)
    query = f"What is the current {topic} decision?"
    budget = rng.randint(420, 560)
    messages: list[dict[str, str]]
    expected = current_value
    excluded: list[str] = []

    if fixture_class == "a":
        target = f"{topic.title()} decision: use {current_value} for the next release."
        messages = [
            {"role": "system", "content": "Retain source-grounded project decisions."},
            {"role": "assistant", "content": target},
            *_chatter(rng, rng.randint(22, 30), offset=2),
        ]
        expected_route = "attention"
    elif fixture_class == "b":
        repeated_token = rng.choice(("cache", "token", "status", "x"))
        repeated = (repeated_token + " ") * rng.randint(430, 520)
        messages = [
            {"role": "assistant", "content": repeated}
            for _ in range(rng.randint(26, 32))
        ]
        expected_route = "raw"
        expected = ""
    elif fixture_class == "c":
        target = f"{topic.title()} decision: use {current_value} for the next release."
        messages = [
            *_chatter(rng, rng.randint(22, 28)),
            {"role": "assistant", "content": target},
        ]
        expected_route = "raw"
    elif fixture_class == "d":
        messages = _chatter(rng, rng.randint(24, 32))
        query = rng.choice((
            "What stands out?", "Any broad thoughts?", "How is everything going?",
        ))
        expected_route = "raw"
        expected = ""
    else:
        old = f"{topic.title()} decision: use {old_value} for the next release."
        current = f"Current {topic} decision: use {current_value} for the next release."
        traceback = (
            "Traceback (most recent call last):\n"
            "  File \"/workspace/router.py\", line 91, in compile_context\n"
            "    raise RuntimeError('temporary routing failure')\n"
            "RuntimeError: temporary routing failure"
        )
        messages = [
            {"role": "assistant", "content": old},
            *_chatter(rng, rng.randint(8, 12), offset=1),
            {"role": "assistant", "content": current},
            {"role": "tool", "content": traceback},
            *_chatter(rng, rng.randint(12, 16), offset=1),
        ]
        expected_route = "attention"
        excluded = [old_value, "Traceback (most recent call last)", "router.py"]

    content_chars = sum(len(message["content"]) for message in messages)
    return {
        "id": case_id,
        "fixture_class": fixture_class,
        "messages": messages,
        "query": query,
        "output_budget_chars": budget,
        "history_content_chars": content_chars,
        "expected_route": expected_route,
        "expected": expected,
        "excluded": excluded,
    }


def generate_class_cases(
    seed: int,
    fixture_class: str,
    n_cases: int = 20,
) -> list[dict[str, Any]]:
    if n_cases < 0:
        raise ValueError("n_cases must be non-negative")
    if fixture_class not in FIXTURE_CLASSES:
        raise ValueError(f"unknown fixture class: {fixture_class}")
    rng = random.Random(f"v14:{seed}:{fixture_class}")
    return [
        generate_fixture(rng, fixture_class, f"v14-{seed}-{fixture_class}-{index:02d}")
        for index in range(n_cases)
    ]


def generate_cases(seed: int, n_cases_per_class: int = 20) -> list[dict[str, Any]]:
    """Generate ``n_cases_per_class`` cases for each frozen fixture class."""
    return [
        case
        for fixture_class in FIXTURE_CLASSES
        for case in generate_class_cases(seed, fixture_class, n_cases_per_class)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-cases-per-class", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cases = generate_cases(args.seed, args.n_cases_per_class)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
