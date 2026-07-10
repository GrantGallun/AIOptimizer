#!/usr/bin/env python3
"""Backend comparison v2: deterministic priority retrieval after v1 cache pollution.

Dev seeds select/falsify the mechanism. Hidden seeds are fixed in this file and
must be run only after the policy and tests are committed. The harness reuses the
frozen v1 records, cases, and scorer unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.backend_comparison_v1 import evaluate_backend
from experiments.brain_runtime.memory_backends import (
    LangGraphMemoryBackend,
    LexicalMemoryBackend,
    NativeMemoryBackend,
    backend_version,
)
from experiments.brain_runtime.memory_backends_v2 import PriorityNativeMemoryBackend


DEV_SEEDS = [11, 23, 37, 41, 59]
HIDDEN_SEEDS = [101, 103, 107, 109, 113]
BACKENDS = ("native_v1", "native_priority_v2", "lexical", "langgraph")


def create_v2_backend(name: str, project_id: str):
    if name == "native_v1":
        return NativeMemoryBackend(project_id)
    if name == "native_priority_v2":
        return PriorityNativeMemoryBackend(project_id)
    if name == "lexical":
        return LexicalMemoryBackend(project_id)
    if name == "langgraph":
        return LangGraphMemoryBackend(project_id)
    raise ValueError(f"unknown v2 backend: {name}")


def run_v2(split: str, *, backend_names: list[str] | None = None) -> dict[str, Any]:
    if split not in {"dev", "hidden"}:
        raise ValueError("split must be 'dev' or 'hidden'")
    seeds = DEV_SEEDS if split == "dev" else HIDDEN_SEEDS
    names = backend_names or list(BACKENDS)
    results = []
    for name in names:
        backend = create_v2_backend(name, f"backend-v2-{split}-{name}")
        result = evaluate_backend(backend, seeds=seeds)
        result["requested_name"] = name
        results.append(result)
    return {
        "benchmark": "brain-runtime-backend-comparison-v2-priority-retrieval",
        "split": split,
        "seeds": seeds,
        "frozen_v1_suite": "backend_comparison_v1.py",
        "design_rationale": "Global frequency is a tie-breaker after relevance, validity, evidence, and recency; it cannot override a better query match.",
        "selected_policy": "native_priority_v2",
        "backend_versions": {"langgraph": backend_version("langgraph")},
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "hidden"], default="dev")
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--out")
    args = parser.parse_args()
    names = [value.strip() for value in args.backends.split(",") if value.strip()]
    payload = run_v2(args.split, backend_names=names)
    out = Path(args.out or f"results/brain_runtime/backend_comparison_v2_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [result["summary"] | {"backend": result["backend"], "requested_name": result["requested_name"]} for result in payload["results"]],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

