#!/usr/bin/env python3
"""Evaluate CoALA within-session learning on recurring novel operators.

Each operator first appears without an available rule. After that answer is graded,
the CoALA arm learns the verified rule into ``PersistentBrainRuntime``. Recurrences
must retrieve the rule before model reasoning. The no-memory control receives the
same problems but performs no retrieval or learning.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.coala import (
    ActionKind,
    CognitiveAction,
    CoALAController,
    DecisionContext,
    LongTermMemoryKind,
)
from experiments.brain_runtime.coala_ollama import OllamaCoALAAdapter
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient


OPERATORS: dict[str, tuple[Callable[[int, int], int], str]] = {
    "glorp": (lambda a, b: a + b + 3, "glorp(a,b) = a + b + 3"),
    "zib": (lambda a, b: a * 2 + b, "zib(a,b) = (a times 2) + b"),
    "frotz": (lambda a, b: a + b * 2, "frotz(a,b) = a + (b times 2)"),
    "quan": (lambda a, b: abs(a - b) + 5, "quan(a,b) = |a - b| + 5"),
    "vlim": (lambda a, b: max(a, b) * 2, "vlim(a,b) = 2 times max(a, b)"),
}
ARMS = ("no_memory", "coala")
DEFAULT_SEED = 20260710
ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*(-?\d+)", re.IGNORECASE)
INTEGER_RE = re.compile(r"-?\d+")
Reasoner = Callable[[str, DecisionContext], str]


def build_sequence(seed: int = DEFAULT_SEED, repetitions: int = 5) -> list[dict[str, Any]]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rng = random.Random(seed)
    names = list(OPERATORS)
    occurrences: list[str] = []
    for _ in range(repetitions):
        round_names = list(names)
        rng.shuffle(round_names)
        occurrences.extend(round_names)

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for step, name in enumerate(occurrences):
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        function, rule = OPERATORS[name]
        first = name not in seen
        seen.add(name)
        rows.append(
            {
                "step": step,
                "operator": name,
                "a": a,
                "b": b,
                "expected": function(a, b),
                "rule": rule,
                "first_appearance": first,
            }
        )
    return rows


def parse_answer(text: str) -> int | None:
    explicit = ANSWER_RE.search(text)
    if explicit:
        return int(explicit.group(1))
    integers = INTEGER_RE.findall(text.strip())
    return int(integers[-1]) if integers else None


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
    use_memory: bool,
) -> tuple[int | None, str, tuple[str, ...], list[str]]:
    instruction = _instruction(problem)

    def policy(context: DecisionContext) -> CognitiveAction:
        if not context.events:
            if use_memory:
                return CognitiveAction.retrieve(
                    f"verified operator rule {problem['operator']}", limit=len(OPERATORS)
                )
            return CognitiveAction.reason(instruction)
        if context.events[-1].action.kind is ActionKind.RETRIEVE:
            expected_key = f"operator:{problem['operator']}"
            context.retrieved = [item for item in context.retrieved if item.key == expected_key]
            context.working["retrieved_memory_ids"] = [item.id for item in context.retrieved]
            return CognitiveAction.reason(instruction)
        if context.events[-1].action.kind is ActionKind.REASON:
            answer = parse_answer(context.events[-1].output)
            return CognitiveAction.ground("answer", value=answer)
        raise RuntimeError("answer policy reached an unexpected state")

    result = controller.run_cycle(
        goal=f"answer {problem['operator']} problem",
        observation=f"{problem['operator']}({problem['a']}, {problem['b']})",
        policy=policy,
    )
    reason_event = next(event for event in result.events if event.action.kind is ActionKind.REASON)
    parsed = parse_answer(reason_event.output)
    return (
        parsed,
        reason_event.output,
        result.retrieved_memory_ids,
        [event.action.kind.value for event in result.events],
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


def evaluate(
    sequence: list[dict[str, Any]],
    *,
    reasoners: dict[str, Reasoner],
    model: str,
    seed: int,
    adapter_metrics: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    if set(reasoners) != set(ARMS):
        raise ValueError(f"reasoners must contain exactly: {', '.join(ARMS)}")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    learned_counts: dict[str, int] = {}

    for arm in ARMS:
        memory = PersistentBrainRuntime()
        controller = CoALAController(
            memory,
            reasoner=reasoners[arm],
            grounding={"answer": lambda arguments, _context: str(arguments.get("value"))},
            max_internal_actions=2,
        )
        learned_operators: set[str] = set()
        for problem in sequence:
            parsed, response, retrieved_ids, event_kinds = _answer_cycle(
                controller,
                problem,
                use_memory=arm == "coala",
            )
            correct = parsed == problem["expected"]
            learned_after_feedback = False
            learned_memory_id = None
            if arm == "coala" and problem["operator"] not in learned_operators:
                learned_memory_id = _learn_verified_rule(controller, problem)
                learned_operators.add(problem["operator"])
                learned_after_feedback = True
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    **problem,
                    "answer": parsed,
                    "response": response,
                    "correct": correct,
                    "retrieved_memory_ids": list(retrieved_ids),
                    "event_kinds": event_kinds,
                    "learned_after_feedback": learned_after_feedback,
                    "learned_memory_id": learned_memory_id,
                }
            )
        learned_counts[arm] = len(learned_operators)

    summaries = [summarize(rows, arm, learned_counts[arm]) for arm in ARMS]
    by_arm = {summary["arm"]: summary for summary in summaries}
    recurrence_gap = by_arm["coala"]["recurrence_accuracy"] - by_arm["no_memory"]["recurrence_accuracy"]
    control_delta = by_arm["no_memory"]["recurrence_accuracy"] - by_arm["no_memory"]["first_appearance_accuracy"]
    return {
        "benchmark": "coala-within-session-learning-v1",
        "model": model,
        "seed": seed,
        "operators": list(OPERATORS),
        "repetitions": len(sequence) // len(OPERATORS),
        "rows": rows,
        "summaries": summaries,
        "adapter_metrics": adapter_metrics or {},
        "gate": {
            "owner": "fable",
            "required_recurrence_accuracy_gap": 0.30,
            "observed_recurrence_accuracy_gap": recurrence_gap,
            "recurrence_gap_condition_met": recurrence_gap >= 0.30,
            "control_first_appearance_accuracy": by_arm["no_memory"]["first_appearance_accuracy"],
            "control_recurrence_accuracy": by_arm["no_memory"]["recurrence_accuracy"],
            "control_recurrence_minus_first_delta": control_delta,
            "note": "Fable interprets the ~= control condition and writes the verdict.",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def summarize(rows: list[dict[str, Any]], arm: str, learned_memories: int) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    first = [row for row in selected if row["first_appearance"]]
    recurrence = [row for row in selected if not row["first_appearance"]]
    return {
        "arm": arm,
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


def run_ollama(*, model: str, seed: int) -> dict[str, Any]:
    client = OllamaClient(timeout_seconds=180.0)
    adapters = {
        arm: OllamaCoALAAdapter(
            client,
            model=model,
            grounding_actions=["answer"],
            max_reason_tokens=96,
            require_retrieval_before_terminal=False,
        )
        for arm in ARMS
    }
    payload = evaluate(
        build_sequence(seed),
        reasoners={arm: adapter.reason for arm, adapter in adapters.items()},
        model=model,
        seed=seed,
    )
    payload["adapter_metrics"] = {arm: adapter.metrics.as_dict() for arm, adapter in adapters.items()}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default="results/brain_runtime/coala_learning.json")
    args = parser.parse_args()
    payload = run_ollama(model=args.model, seed=args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'arm':<12} {'first':>8} {'recurrence':>11} {'learned':>8} {'mean_retrieved':>15}")
    for summary in payload["summaries"]:
        print(
            f"{summary['arm']:<12} {summary['first_appearance_accuracy']:>8.3f} "
            f"{summary['recurrence_accuracy']:>11.3f} {summary['learned_memories']:>8} "
            f"{summary['mean_retrieved_memories']:>15.3f}"
        )
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
