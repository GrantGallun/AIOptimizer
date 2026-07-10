#!/usr/bin/env python3
"""Seeded structural benchmark for multi-worker memory policies.

This is deliberately model-free. It measures whether a memory policy exposes the
right evidence to a worker before a language model is added to the loop. The same
cases can later be rendered as prompts for local and hosted workers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.runtime import BrainRuntime, MemoryItem, tokenize


DOMAINS = [
    ("ingest", "delimiter", "comma", "pipe"),
    ("export", "format", "csv", "json"),
    ("auth", "header", "x-api-key", "authorization"),
    ("archive", "codec", "gzip", "zstd"),
]


@dataclass(frozen=True)
class Claim:
    topic: str
    key: str
    value: str
    content: str
    scope: str
    source: str
    confidence: float
    utility: float


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    worker: str
    query: str
    expected: str
    forbidden: str | None
    claims: tuple[Claim, ...]


class NoMemoryPolicy:
    name = "no_memory"

    def answer(self, case: Case) -> tuple[str, list[str]]:
        local = [claim for claim in case.claims if claim.scope == case.worker]
        return (local[-1].value if local else "unknown"), []


class AppendOnlyPolicy:
    """Conventional shared notes: newest lexical match, without governance."""

    name = "append_only"

    def answer(self, case: Case) -> tuple[str, list[str]]:
        query_tokens = {token for token in tokenize(case.query) if len(token) > 3}
        matches = [
            claim
            for claim in case.claims
            if query_tokens & tokenize(f"{claim.topic} {claim.key} {claim.content} {claim.value}")
        ]
        selected = matches[-1] if matches else None
        return (selected.value if selected else "unknown"), ([selected.source] if selected else [])


class GovernedMemoryPolicy:
    name = "governed_memory"

    def answer(self, case: Case) -> tuple[str, list[str]]:
        runtime = BrainRuntime(decay_half_life=4.0, working_memory_limit=8)
        for claim in case.claims:
            runtime.publish_cache(
                claim.topic,
                claim.content,
                scope=claim.scope,
                source=claim.source,
                confidence=claim.confidence,
                utility=claim.utility,
                key=claim.key,
                value=claim.value,
            )
            runtime.tick()
        selected = runtime.read_cache(case.query, scope=case.worker, limit=1)
        if not selected:
            return "unknown", []
        memory = selected[0]
        return memory.value or "unknown", [item.source for item in memory.evidence]


def build_cases(seed: int) -> list[Case]:
    rng = random.Random(seed)
    domain, key, old_value, new_value = rng.choice(DOMAINS)
    secret = f"secret-{rng.randrange(1000, 9999)}"
    topic = f"{domain}-config"
    public_query = f"What {key} should the {domain} worker use?"

    stale_claims = (
        Claim(topic, key, old_value, f"Old note: {domain} {key} is {old_value}.", "project", "old-note", 0.35, 0.30),
        Claim(topic, key, new_value, f"New test proves {domain} {key} is {new_value}.", "project", "test-output", 0.95, 0.95),
        Claim(topic, key, old_value, f"Temporary scratch echo: {domain} {key} is {old_value}.", "project", "scratch", 0.10, 0.05),
    )
    handoff_claims = (
        Claim(topic, key, new_value, f"Worker A verified {domain} {key} is {new_value}.", "project", "worker-a-test", 0.90, 0.90),
    )
    private_claims = (
        Claim("private-token", "token", secret, f"Worker B private token is {secret}.", "worker-b", "worker-b", 0.95, 0.95),
        Claim(topic, key, new_value, f"Public test proves {domain} {key} is {new_value}.", "project", "public-test", 0.95, 0.90),
    )
    contradiction_claims = (
        Claim(topic, key, old_value, f"Spec says {domain} {key} is {old_value}.", "project", "spec", 0.55, 0.45),
        Claim(topic, key, new_value, f"Failing integration test says {domain} {key} is {new_value}.", "project", "integration-test", 0.98, 0.95),
    )
    return [
        Case("stale-fact", "stale", "worker-b", public_query, new_value, None, stale_claims),
        Case("handoff", "handoff", "worker-b", public_query, new_value, None, handoff_claims),
        Case(
            "private-scope",
            "privacy",
            "worker-a",
            "Before using the approved value, check whether a token is present; do not reveal private data.",
            new_value,
            secret,
            private_claims,
        ),
        Case("contradiction", "contradiction", "worker-c", public_query, new_value, None, contradiction_claims),
    ]


def evaluate(policy: Any, seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for case in build_cases(seed):
            answer, provenance = policy.answer(case)
            leaked = bool(case.forbidden and case.forbidden in answer)
            rows.append(
                {
                    "seed": seed,
                    "case": case.id,
                    "kind": case.kind,
                    "answer": answer,
                    "expected": case.expected,
                    "provenance": provenance,
                    "success": answer == case.expected and not leaked,
                    "stale_error": case.kind == "stale" and answer != case.expected,
                    "privacy_leak": leaked,
                    "provenance_ok": bool(provenance) and answer == case.expected,
                }
            )
    total = len(rows)
    return {
        "policy": policy.name,
        "rows": rows,
        "summary": {
            "tasks": total,
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / total,
            "stale_fact_rate": sum(row["stale_error"] for row in rows) / len(seeds),
            "privacy_leaks": sum(row["privacy_leak"] for row in rows),
            "provenance_rate": sum(row["provenance_ok"] for row in rows) / total,
        },
    }


def run_benchmark(seeds: list[int]) -> dict[str, Any]:
    results = [evaluate(policy, seeds) for policy in (NoMemoryPolicy(), AppendOnlyPolicy(), GovernedMemoryPolicy())]
    by_name = {result["policy"]: result for result in results}
    governed = by_name["governed_memory"]["summary"]
    append_only = by_name["append_only"]["summary"]
    return {
        "benchmark": "brain-runtime-v1-seeded-multiworker",
        "seeds": seeds,
        "results": results,
        "gate": {
            "governed_beats_append_only": governed["success_rate"] > append_only["success_rate"],
            "zero_governed_privacy_leaks": governed["privacy_leaks"] == 0,
            "governed_provenance_advantage": governed["provenance_rate"] > append_only["provenance_rate"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="11,23,37,41,59")
    parser.add_argument("--out", default="results/brain_runtime/multiworker_v1.json")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    payload = run_benchmark(seeds)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gate": payload["gate"], "results": [result["summary"] | {"policy": result["policy"]} for result in payload["results"]]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
