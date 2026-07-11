#!/usr/bin/env python3
"""Evaluate CoALA learning with a large, procedurally generated rule memory.

This is a scale extension of ``coala_learning_eval.py``.  It keeps the
within-session learning protocol fixed while comparing BrainRuntime's native
Jaccard retrieval with encoder-ranked top-k retrieval.  The gate is deliberately
reported as a component for Fable to interpret; this module does not write a
research verdict.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.context_selection import make_operators, select_topk
from experiments.brain_runtime.coala import (
    ActionKind,
    CognitiveAction,
    CoALAController,
    DecisionContext,
    LongTermMemoryKind,
)
from experiments.brain_runtime.coala_ollama import OllamaCoALAAdapter
from experiments.brain_runtime.runtime import MemoryItem
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient


DEFAULT_SEED = 20260710
DEFAULT_OPERATOR_COUNT = 30
DEFAULT_REPETITIONS = 5
DEFAULT_TOP_K = 3
RETRIEVAL_MODES = ("jaccard", "encoder")
Reasoner = Callable[[str, DecisionContext], str]
Selector = Callable[[str, list[str], int], list[str]]


class EncoderMemoryRuntime(PersistentBrainRuntime):
    """Persistent runtime whose retrieval ranking is encoder-selected top-k."""

    def __init__(self, *, selector: Selector = select_topk) -> None:
        super().__init__()
        self.selector = selector

    def retrieve(self, query: str, *, scope: str = "project", limit: int = 5) -> list[MemoryItem]:
        if limit < 1:
            raise ValueError("retrieval limit must be positive")
        visible = [
            item
            for item in [
                *self.working_memory.values(),
                *self.long_term_memory.values(),
                *self.shared_cache.values(),
            ]
            if item.scope in {scope, "project", "global"}
        ]
        if len(visible) <= limit:
            selected = visible
        else:
            # Include the topic and structured value so the selector sees the
            # same operator identity that the native runtime indexes.
            candidates = [
                f"{item.topic}\n{item.content}\n{item.key or ''}\n{item.value or ''}"
                for item in visible
            ]
            selected_texts = self.selector(query, candidates, limit)
            by_text = {text: item for text, item in zip(candidates, visible)}
            selected = [by_text[text] for text in selected_texts if text in by_text]

        for item in selected:
            item.uses += 1
            item.last_accessed = self.clock
        return selected


def build_sequence(
    operators: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    repetitions: int = DEFAULT_REPETITIONS,
) -> list[dict[str, Any]]:
    """Build balanced, interleaved occurrences with first-appearance flags."""
    if not operators:
        raise ValueError("operators must not be empty")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    names = [str(operator["name"]) for operator in operators]
    if len(names) != len(set(names)):
        raise ValueError("operator names must be unique")
    by_name = {str(operator["name"]): operator for operator in operators}
    rng = random.Random(seed)
    occurrences: list[str] = []
    for _ in range(repetitions):
        round_names = list(names)
        rng.shuffle(round_names)
        occurrences.extend(round_names)

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for step, name in enumerate(occurrences):
        operator = by_name[name]
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        rows.append(
            {
                "step": step,
                "operator": name,
                "a": a,
                "b": b,
                "expected": int(operator["fn"](a, b)),
                "rule": str(operator["rule_text"]),
                "first_appearance": name not in seen,
            }
        )
        seen.add(name)
    return rows


def _instruction(problem: dict[str, Any]) -> str:
    return (
        "Solve the novel-operator problem. Apply an authorized retrieved rule if one is present. "
        "Do not invent a rule when none is present. End with exactly ANSWER=<integer>. "
        f"Problem: {problem['operator']}({problem['a']}, {problem['b']})"
    )


def _answer_cycle(
    controller: CoALAController,
    problem: dict[str, Any],
    *,
    retrieval_limit: int,
) -> tuple[int | None, str, tuple[str, ...], tuple[str, ...]]:
    instruction = _instruction(problem)

    def policy(context: DecisionContext) -> CognitiveAction:
        if not context.events:
            return CognitiveAction.retrieve(
                f"operator:{problem['operator']}", limit=retrieval_limit
            )
        if context.events[-1].action.kind is ActionKind.RETRIEVE:
            expected_key = f"operator:{problem['operator']}"
            context.retrieved = [item for item in context.retrieved if item.key == expected_key]
            context.working["retrieved_memory_ids"] = [item.id for item in context.retrieved]
            return CognitiveAction.reason(instruction)
        if context.events[-1].action.kind is ActionKind.REASON:
            from experiments.brain_runtime.coala_learning_eval import parse_answer

            answer = parse_answer(context.events[-1].output)
            return CognitiveAction.ground("answer", value=answer)
        raise RuntimeError("answer policy reached an unexpected state")

    result = controller.run_cycle(
        goal=f"answer {problem['operator']} problem",
        observation=f"{problem['operator']}({problem['a']}, {problem['b']})",
        policy=policy,
    )
    reason_event = next(event for event in result.events if event.action.kind is ActionKind.REASON)
    from experiments.brain_runtime.coala_learning_eval import parse_answer

    return (
        parse_answer(reason_event.output),
        reason_event.output,
        result.retrieved_memory_ids,
        tuple(event.action.kind.value for event in result.events),
    )


def _learn_verified_rule(controller: CoALAController, problem: dict[str, Any]) -> str:
    rule = str(problem["rule"])

    def policy(_context: DecisionContext) -> CognitiveAction:
        return CognitiveAction.learn(
            LongTermMemoryKind.SEMANTIC,
            topic=f"operator:{problem['operator']}",
            content=f"Verified operator rule: {rule}",
            key=f"operator:{problem['operator']}",
            value=rule,
            confidence=1.0,
            utility=1.0,
            source="graded-operator-feedback",
        )

    result = controller.run_cycle(
        goal=f"learn verified rule for {problem['operator']}",
        observation=f"Feedback after grading: {rule}",
        policy=policy,
    )
    return result.output


def _runtime(mode: str) -> PersistentBrainRuntime:
    if mode == "jaccard":
        return PersistentBrainRuntime()
    if mode == "encoder":
        return EncoderMemoryRuntime()
    raise ValueError(f"unknown retrieval mode: {mode}")


def evaluate(
    sequence: list[dict[str, Any]],
    *,
    reasoners: dict[str, Reasoner],
    model: str,
    seed: int,
    retrieval_limit: int = DEFAULT_TOP_K,
    runtimes: dict[str, PersistentBrainRuntime] | None = None,
) -> dict[str, Any]:
    """Run both retrieval modes with identical problems and learning rules."""
    if set(reasoners) != set(RETRIEVAL_MODES):
        raise ValueError(f"reasoners must contain exactly: {', '.join(RETRIEVAL_MODES)}")
    if retrieval_limit < 1:
        raise ValueError("retrieval_limit must be positive")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    learned_counts: dict[str, int] = {}

    for mode in RETRIEVAL_MODES:
        memory = (runtimes or {}).get(mode, _runtime(mode))
        controller = CoALAController(
            memory,
            reasoner=reasoners[mode],
            grounding={"answer": lambda arguments, _context: str(arguments.get("value"))},
            max_internal_actions=2,
        )
        learned_operators: set[str] = set()
        for problem in sequence:
            answer, response, retrieved_ids, event_kinds = _answer_cycle(
                controller, problem, retrieval_limit=retrieval_limit
            )
            correct = answer == problem["expected"]
            learned_after_feedback = problem["operator"] not in learned_operators
            learned_memory_id = None
            if learned_after_feedback:
                learned_memory_id = _learn_verified_rule(controller, problem)
                learned_operators.add(problem["operator"])
            rows.append(
                {
                    "seed": seed,
                    "mode": mode,
                    **problem,
                    "answer": answer,
                    "response": response,
                    "correct": correct,
                    "retrieved_memory_ids": list(retrieved_ids),
                    "event_kinds": list(event_kinds),
                    "learned_after_feedback": learned_after_feedback,
                    "learned_memory_id": learned_memory_id,
                }
            )
        learned_counts[mode] = len(learned_operators)

    summaries = [summarize(rows, mode, learned_counts[mode]) for mode in RETRIEVAL_MODES]
    by_mode = {summary["mode"]: summary for summary in summaries}
    recurrence_gap = (
        by_mode["encoder"]["recurrence_accuracy"]
        - by_mode["jaccard"]["recurrence_accuracy"]
    )
    return {
        "benchmark": "coala-within-session-learning-scale-v1",
        "model": model,
        "seed": seed,
        "operator_count": len({row["operator"] for row in sequence}),
        "repetitions": len(sequence) // len({row["operator"] for row in sequence}),
        "retrieval_limit": retrieval_limit,
        "retrieval_modes": list(RETRIEVAL_MODES),
        "rows": rows,
        "summaries": summaries,
        "adapter_metrics": {},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "gate": {
            "owner": "fable",
            "required_recurrence_accuracy_gap": 0.20,
            "observed_recurrence_accuracy_gap": recurrence_gap,
            "recurrence_gap_condition_met": recurrence_gap >= 0.20,
            "note": "Fable interprets the scale retrieval gate and writes the verdict.",
        },
    }


def summarize(rows: list[dict[str, Any]], mode: str, learned_memories: int) -> dict[str, Any]:
    selected = [row for row in rows if row["mode"] == mode]
    first = [row for row in selected if row["first_appearance"]]
    recurrence = [row for row in selected if not row["first_appearance"]]
    return {
        "mode": mode,
        "tasks": len(selected),
        "first_appearance_tasks": len(first),
        "recurrence_tasks": len(recurrence),
        "first_appearance_correct": sum(row["correct"] for row in first),
        "recurrence_correct": sum(row["correct"] for row in recurrence),
        "first_appearance_accuracy": _accuracy(first),
        "recurrence_accuracy": _accuracy(recurrence),
        "learned_memories": learned_memories,
        "mean_retrieved_memories": sum(len(row["retrieved_memory_ids"]) for row in selected) / len(selected),
    }


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(row["correct"] for row in rows) / len(rows) if rows else 0.0


def run_ollama(
    *,
    model: str,
    seed: int,
    operator_count: int = DEFAULT_OPERATOR_COUNT,
    repetitions: int = DEFAULT_REPETITIONS,
    retrieval_limit: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    client = OllamaClient(timeout_seconds=180.0)
    operators = make_operators(operator_count, seed=seed)
    sequence = build_sequence(operators, seed=seed, repetitions=repetitions)
    adapters = {
        mode: OllamaCoALAAdapter(
            client,
            model=model,
            grounding_actions=["answer"],
            max_reason_tokens=96,
            require_retrieval_before_terminal=False,
        )
        for mode in RETRIEVAL_MODES
    }
    payload = evaluate(
        sequence,
        reasoners={mode: adapter.reason for mode, adapter in adapters.items()},
        model=model,
        seed=seed,
        retrieval_limit=retrieval_limit,
    )
    payload["adapter_metrics"] = {mode: adapter.metrics.as_dict() for mode, adapter in adapters.items()}
    payload["operators"] = [
        {"name": operator["name"], "rule_text": operator["rule_text"]} for operator in operators
    ]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-operators", type=int, default=DEFAULT_OPERATOR_COUNT)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", default="results/brain_runtime/coala_scale.json")
    args = parser.parse_args()
    payload = run_ollama(
        model=args.model,
        seed=args.seed,
        operator_count=args.n_operators,
        repetitions=args.repetitions,
        retrieval_limit=args.top_k,
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'mode':<12} {'first':>8} {'recurrence':>11} {'learned':>8} {'mean_retrieved':>15}")
    for summary in payload["summaries"]:
        print(
            f"{summary['mode']:<12} {summary['first_appearance_accuracy']:>8.3f} "
            f"{summary['recurrence_accuracy']:>11.3f} {summary['learned_memories']:>8} "
            f"{summary['mean_retrieved_memories']:>15.3f}"
        )
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
