#!/usr/bin/env python3
"""Synthetic Brain Runtime v0 benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.runtime import BrainRuntime, VanillaLoop


def run_benchmark() -> dict[str, Any]:
    started = time.perf_counter()
    brain = BrainRuntime(decay_half_life=4.0, working_memory_limit=4)
    vanilla = VanillaLoop()

    for system in [brain, vanilla]:
        system.remember(
            "ingest-parser",
            "The ingest parser delimiter is comma.",
            key="delimiter",
            value="comma",
            confidence=0.45,
            utility=0.4,
            source="old-note",
        )
        system.remember(
            "api-auth",
            "API auth accepts token header v1.",
            key="auth_header",
            value="v1",
            confidence=0.4,
            utility=0.4,
            source="old-note",
        )

    brain.tick(6)
    brain.publish_cache(
        "ingest-parser",
        "New failing test proves the ingest parser delimiter is pipe.",
        scope="project",
        source="test-output",
        key="delimiter",
        value="pipe",
        confidence=0.95,
        utility=0.9,
    )
    brain.publish_cache(
        "json-parser",
        "Regex extraction failed on nested payload extraction; use structured JSON parsing.",
        scope="project",
        source="hypothesis-test",
        confidence=0.9,
        utility=0.95,
    )
    vanilla.remember(
        "ingest-parser",
        "New failing test proves the ingest parser delimiter is pipe.",
        key="delimiter",
        value="pipe",
        confidence=0.95,
        utility=0.9,
        source="test-output",
    )
    vanilla.remember(
        "json-parser",
        "Regex extraction failed on nested payload extraction; use structured JSON parsing.",
        confidence=0.9,
        utility=0.95,
        source="hypothesis-test",
    )

    brain.remember(
        "distractor",
        "Old speculation: prefer comma-separated scratch files for temporary notes.",
        confidence=0.2,
        utility=0.1,
        source="scratch",
    )
    vanilla.remember(
        "distractor",
        "Old speculation: prefer comma-separated scratch files for temporary notes.",
        confidence=0.2,
        utility=0.1,
        source="scratch",
    )
    brain.tick(20)
    consolidation = brain.consolidate()
    brain.propose_tasks("ingest parser delimiter conflict")

    tasks = [
        {
            "id": "stale-fact",
            "query": "What delimiter should the ingest parser use?",
            "expected": "pipe",
            "brain_answer": answer_value(brain.retrieve("ingest parser delimiter", limit=1)),
            "vanilla_answer": answer_value(vanilla.retrieve("ingest parser delimiter", limit=1)),
        },
        {
            "id": "transfer",
            "query": "How should nested payload extraction be implemented?",
            "expected": "structured JSON parsing",
            "brain_answer": answer_content(brain.retrieve("nested payload extraction parser", limit=1)),
            "vanilla_answer": answer_content(vanilla.retrieve("nested payload extraction parser", limit=1)),
        },
        {
            "id": "forgetting",
            "query": "Should temporary scratch comma notes affect parser decisions?",
            "expected": "no distractor",
            "brain_answer": answer_content(brain.retrieve("temporary scratch comma notes", limit=1)),
            "vanilla_answer": answer_content(vanilla.retrieve("temporary scratch comma notes", limit=1)),
        },
        {
            "id": "task-hook",
            "query": "Did contradiction pressure spawn a resolution task?",
            "expected": "task",
            "brain_answer": "task" if brain.task_hooks else "none",
            "vanilla_answer": "none",
        },
    ]

    for task in tasks:
        task["brain_success"] = score(task["id"], task["brain_answer"], task["expected"])
        task["vanilla_success"] = score(task["id"], task["vanilla_answer"], task["expected"])

    brain_successes = sum(1 for task in tasks if task["brain_success"])
    vanilla_successes = sum(1 for task in tasks if task["vanilla_success"])
    payload = {
        "benchmark": "brain-runtime-v0-synthetic",
        "summary": {
            "tasks": len(tasks),
            "brain_successes": brain_successes,
            "vanilla_successes": vanilla_successes,
            "brain_success_rate": brain_successes / len(tasks),
            "vanilla_success_rate": vanilla_successes / len(tasks),
            "brain_beats_vanilla": brain_successes > vanilla_successes,
            "contradictions": len(brain.contradictions),
            "task_hooks": len(brain.task_hooks),
            "consolidation": consolidation,
        },
        "tasks": tasks,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    return payload


def answer_value(memories: list[Any]) -> str:
    if not memories:
        return ""
    return getattr(memories[0], "value", None) or memories[0].get("value", "")


def answer_content(memories: list[Any]) -> str:
    if not memories:
        return ""
    return getattr(memories[0], "content", None) or memories[0].get("content", "")


def score(task_id: str, answer: str, expected: str) -> bool:
    if task_id == "forgetting":
        lowered = answer.lower()
        return "distractor" not in lowered and "scratch" not in lowered and "comma" not in lowered
    return expected.lower() in answer.lower()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/brain_runtime/benchmark_v0.json")
    args = parser.parse_args()
    payload = run_benchmark()
    write_json(Path(args.out), payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
