#!/usr/bin/env python3
"""Retrieval-presentation ablation for governed memory (PREREGISTRATION_v2).

Same governed `read_cache` retrieval as v1 every time; only the rendering
shown to the model changes. This module reuses the v1 case generator, prompt
builder, and scorer unchanged (`multiworker_benchmark.build_cases`,
`local_worker_eval.prompt_for`/`score_response`/`context_for`/`summarize`) and
adds only the render functions and governed-context builder needed to test
whether presentation, not retrieval, was the v1 bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.local_worker_eval import context_for, prompt_for, score_response, summarize
from experiments.brain_runtime.multiworker_benchmark import Case, build_cases
from experiments.brain_runtime.runtime import BrainRuntime, MemoryItem
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient


RenderFn = Callable[[list[MemoryItem]], str]

NO_MEMORIES_TEXT = "No relevant shared notes are available."

REFERENCE_ARMS: tuple[str, ...] = ("no_memory", "append_only")


def render_v1(memories: list[MemoryItem]) -> str:
    """Exact v1 format: `- [source] content` lines, best-ranked first."""
    if not memories:
        return NO_MEMORIES_TEXT
    return "\n".join(f"- [{memory.evidence[0].source}] {memory.content}" for memory in memories)


def render_value_forward(memories: list[MemoryItem]) -> str:
    """States the resolved value plainly; the source is a trailing parenthetical."""
    if not memories:
        return NO_MEMORIES_TEXT
    memory = memories[0]
    return f"The verified {memory.key} is {memory.value}. (source: {memory.evidence[0].source})"


CONDITIONS = {
    "v1_repro": (2, render_v1),
    "resolved_only": (1, render_v1),
    "value_forward": (1, render_value_forward),
}

ARMS: tuple[str, ...] = REFERENCE_ARMS + tuple(CONDITIONS.keys())


def governed_context(case: Case, *, limit: int, render_fn: RenderFn) -> tuple[str, list[str]]:
    """Replicates the governed branch of v1 `context_for`, varying only limit/render."""
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
    memories = runtime.read_cache(case.query, scope=case.worker, limit=limit)
    sources = [evidence.source for memory in memories for evidence in memory.evidence]
    return render_fn(memories), sources


def build_context(case: Case, arm: str) -> tuple[str, list[str]]:
    if arm in REFERENCE_ARMS:
        return context_for(case, arm)
    limit, render_fn = CONDITIONS[arm]
    return governed_context(case, limit=limit, render_fn=render_fn)


def run_ablation(
    client: OllamaClient | None,
    *,
    model: str = DEFAULT_MODEL,
    dev_seeds: list[int],
    hidden_seeds: list[int],
    arms: tuple[str, ...] = ARMS,
    render_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    splits = {"dev": dev_seeds, "hidden": hidden_seeds}
    for split, seeds in splits.items():
        for seed in seeds:
            for case in build_cases(seed):
                for arm in arms:
                    notes, provenance = build_context(case, arm)
                    prompt = prompt_for(case, notes)
                    if render_only:
                        response_text, prompt_tokens, completion_tokens, total_duration_ns = "", 0, 0, 0
                    else:
                        generation = client.generate_with_metrics(prompt, model=model)
                        response_text = generation.text
                        prompt_tokens = generation.prompt_tokens
                        completion_tokens = generation.completion_tokens
                        total_duration_ns = generation.total_duration_ns
                    score = score_response(response_text, case)
                    rows.append(
                        {
                            "seed": seed,
                            "case": case.id,
                            "kind": case.kind,
                            "arm": arm,
                            "policy": arm,
                            "expected": case.expected,
                            "forbidden": case.forbidden,
                            "response": response_text,
                            "provenance": provenance,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_duration_ns": total_duration_ns,
                            "split": split,
                            "prompt": prompt,
                            **score,
                        }
                    )
    summaries = [
        {**summarize([row for row in rows if row["split"] == split], arm), "split": split}
        for split in ("dev", "hidden")
        for arm in arms
    ]
    gate = compute_gate(rows)
    return {
        "benchmark": "brain-runtime-v2-presentation-ablation",
        "model": model,
        "dev_seeds": dev_seeds,
        "hidden_seeds": hidden_seeds,
        "arms": list(arms),
        "render_only": render_only,
        "rows": rows,
        "summaries": summaries,
        "gate": gate,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def compute_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Decision gate per PREREGISTRATION_v2: C2 (value_forward) vs append_only, hidden seeds."""
    hidden = [row for row in rows if row["split"] == "hidden"]
    value_forward = summarize(hidden, "value_forward")
    append_only = summarize(hidden, "append_only")
    value_forward_contradiction_successes = sum(
        row["success"] for row in hidden if row["policy"] == "value_forward" and row["kind"] == "contradiction"
    )
    append_only_contradiction_successes = sum(
        row["success"] for row in hidden if row["policy"] == "append_only" and row["kind"] == "contradiction"
    )
    gate = {
        "governed_success_ge_append_only": value_forward["successes"] >= append_only["successes"],
        "zero_privacy_leaks": value_forward["privacy_leaks"] == 0,
        "zero_stale_errors": value_forward["stale_errors"] == 0,
        "contradiction_success_ge_append_only": value_forward_contradiction_successes
        >= append_only_contradiction_successes,
    }
    gate["passed"] = all(gate.values())
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dev-seeds", default="11,23,37,41,59")
    parser.add_argument("--hidden-seeds", default="101,103,107,109,113")
    parser.add_argument("--out", default="results/brain_runtime/presentation_ablation_v2.json")
    parser.add_argument("--render-only", action="store_true", help="Build prompts without calling Ollama.")
    args = parser.parse_args()
    dev_seeds = [int(value) for value in args.dev_seeds.split(",") if value]
    hidden_seeds = [int(value) for value in args.hidden_seeds.split(",") if value]
    client = None if args.render_only else OllamaClient(timeout_seconds=180.0)
    payload = run_ablation(
        client,
        model=args.model,
        dev_seeds=dev_seeds,
        hidden_seeds=hidden_seeds,
        render_only=args.render_only,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hidden_summaries = [summary for summary in payload["summaries"] if summary["split"] == "hidden"]
    print(json.dumps(hidden_summaries, indent=2, sort_keys=True))
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
