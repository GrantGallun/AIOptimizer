#!/usr/bin/env python3
"""Exploratory cross-session memory-backend comparison v1.

This is a backend plumbing and structural retrieval harness, not a preregistered
model-quality verdict. It holds records, scopes, queries, and scoring constant
while swapping native AIOptimizer, a lexical baseline, LangGraph, Mem0, or
Graphiti behind the common contract in ``memory_backends.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.memory_backends import (
    BackendUnavailable,
    MemoryBackend,
    MemoryRecord,
    backend_version,
    create_backend,
)


DOMAINS = (
    ("parser", "delimiter", "comma", "pipe"),
    ("export", "format", "csv", "json"),
    ("archive", "codec", "gzip", "zstd"),
    ("auth", "header", "x-api-key", "authorization"),
)


@dataclass(frozen=True)
class ComparisonCase:
    id: str
    kind: str
    query: str
    scope: str
    expected_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]


def build_records_and_cases(seed: int) -> tuple[list[MemoryRecord], list[ComparisonCase]]:
    rng = random.Random(seed)
    domain, key, old, new = rng.choice(DOMAINS)
    secret = f"secret-{rng.randrange(1000, 9999)}"
    prefix = f"s{seed}"
    records = [
        MemoryRecord(f"{prefix}-old", f"{prefix}-{domain}-config", f"Scenario {prefix} old specification: {domain} {key} is {old}.", "project", "old-spec", key=key, value=old, confidence=0.35, utility=0.3, observed_at=1),
        MemoryRecord(f"{prefix}-new", f"{prefix}-{domain}-config", f"Scenario {prefix} verified integration test: {domain} {key} is {new}.", "project", "integration-test", key=key, value=new, confidence=0.98, utility=0.95, observed_at=2),
        MemoryRecord(f"{prefix}-scratch", f"{prefix}-{domain}-scratch", f"Scenario {prefix} unverified scratch note repeats {old} for an unrelated prototype.", "project", "scratch", confidence=0.1, utility=0.05, observed_at=3),
        MemoryRecord(f"{prefix}-episode", f"{prefix}-nested-payload-failure", f"Scenario {prefix} previous attempt failed because regex parsed nested payloads incorrectly; the successful fix used structured JSON parsing.", "project", "test-retrospective", kind="episodic", confidence=0.95, utility=0.9, observed_at=4),
        MemoryRecord(f"{prefix}-procedure", f"{prefix}-project-test-procedure", f"For scenario {prefix}, validate changes with python -m unittest discover -s tests before commit.", "project", "verified-procedure", kind="procedural", confidence=0.95, utility=0.95, observed_at=5),
        MemoryRecord(f"{prefix}-private", f"{prefix}-private-token", f"Scenario {prefix} Worker B private deployment token is {secret}.", "worker-b", "worker-b", confidence=0.99, utility=0.9, observed_at=6),
        MemoryRecord(f"{prefix}-public", f"{prefix}-deployment-region", f"Scenario {prefix} approved public deployment region is us-central.", "project", "deployment-test", confidence=0.95, utility=0.85, observed_at=7),
        MemoryRecord(f"{prefix}-noise", f"{prefix}-frontend-theme", f"Scenario {prefix} dashboard accent color is violet.", "project", "design-note", confidence=0.7, utility=0.2, observed_at=8),
    ]
    cases = [
        ComparisonCase("stale", "stale", f"For scenario {prefix}, what {key} should the {domain} use according to the verified integration test?", "worker-a", (new,), (old,)),
        ComparisonCase("episodic-transfer", "episodic", f"For scenario {prefix}, how should nested payload parsing avoid the previous failed approach?", "worker-a", ("structured", "json"), ()),
        ComparisonCase("procedure", "procedural", f"For scenario {prefix}, what command validates this project before commit?", "worker-a", ("python", "unittest", "discover"), ()),
        ComparisonCase("privacy", "privacy", f"For scenario {prefix}, which public deployment region is approved? Do not expose private worker data.", "worker-a", ("us-central",), (secret,)),
    ]
    return records, cases


def evaluate_backend(
    backend: MemoryBackend,
    *,
    seeds: list[int],
    limit: int = 3,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for seed in seeds:
            records, cases = build_records_and_cases(seed)
            backend.begin_session(f"seed-{seed}-write")
            for record in records:
                backend.add(record)
            backend.begin_session(f"seed-{seed}-read")
            for case in cases:
                query_started = time.perf_counter()
                hits = backend.search(case.query, scope=case.scope, limit=limit)
                latency = time.perf_counter() - query_started
                top_content = hits[0].record.content if hits else ""
                all_content = "\n".join(hit.record.content for hit in hits)
                expected = all(term.lower() in top_content.lower() for term in case.expected_terms)
                top_forbidden = any(term.lower() in top_content.lower() for term in case.forbidden_terms)
                leaked = case.kind == "privacy" and any(
                    term.lower() in all_content.lower() for term in case.forbidden_terms
                )
                success = expected and not top_forbidden
                rows.append(
                    {
                        "seed": seed,
                        "case": case.id,
                        "kind": case.kind,
                        "query": case.query,
                        "scope": case.scope,
                        "expected_terms": list(case.expected_terms),
                        "forbidden_terms": list(case.forbidden_terms),
                        "hits": [
                            {
                                "id": hit.record.id,
                                "content": hit.record.content,
                                "scope": hit.record.scope,
                                "source": hit.record.source,
                                "score": hit.score,
                                "backend_metadata": hit.backend_metadata,
                            }
                            for hit in hits
                        ],
                        "success": success and not leaked,
                        "privacy_leak": leaked,
                        "stale_error": case.kind == "stale" and not success,
                        "latency_seconds": round(latency, 6),
                    }
                )
    finally:
        backend.close()
    return {
        "backend": backend.name,
        "rows": rows,
        "summary": {
            "tasks": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else 0.0,
            "privacy_leaks": sum(row["privacy_leak"] for row in rows),
            "stale_errors": sum(row["stale_error"] for row in rows),
            "mean_query_latency_seconds": round(
                sum(row["latency_seconds"] for row in rows) / len(rows), 6
            ) if rows else 0.0,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        },
    }


def run_comparison(
    backend_names: list[str],
    *,
    seeds: list[int],
    mem0_config: str | Path | None = None,
    backend_factory: Callable[..., MemoryBackend] = create_backend,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for name in backend_names:
        project_id = f"backend-v1-{name}-{'-'.join(map(str, seeds))}"
        try:
            backend = backend_factory(name, project_id, mem0_config=mem0_config)
        except BackendUnavailable as exc:
            availability.append({"backend": name, "available": False, "reason": str(exc), "version": backend_version(name)})
            continue
        availability.append({"backend": name, "available": True, "reason": "", "version": backend_version(name)})
        results.append(evaluate_backend(backend, seeds=seeds))
    return {
        "benchmark": "brain-runtime-backend-comparison-v1-exploratory",
        "preregistered": False,
        "model_calls": False,
        "seeds": seeds,
        "requested_backends": backend_names,
        "availability": availability,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="native,lexical,langgraph,mem0,graphiti")
    parser.add_argument("--seeds", default="11,23,37,41,59")
    parser.add_argument("--mem0-config")
    parser.add_argument("--out", default="results/brain_runtime/backend_comparison_v1.json")
    args = parser.parse_args()
    backend_names = [value.strip() for value in args.backends.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    payload = run_comparison(backend_names, seeds=seeds, mem0_config=args.mem0_config)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"availability": payload["availability"], "summaries": [result["summary"] | {"backend": result["backend"]} for result in payload["results"]]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
