#!/usr/bin/env python3
"""Evaluate free-form versus schema-constrained CoALA policy actions.

The problem sequence and grading are reused from ``coala_learning_eval``.  A
``--render-only`` run uses a deterministic local client, which exercises the
same adapter/controller path without requiring an Ollama service.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.coala import (
    ActionKind,
    CognitiveAction,
    CoALAController,
    CycleLimitExceeded,
    LongTermMemoryKind,
)
from experiments.brain_runtime.coala_learning_eval import (
    DEFAULT_SEED,
    OPERATORS,
    build_sequence,
    parse_answer,
)
from experiments.brain_runtime.coala_ollama import OllamaCoALAAdapter
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime
from experiments.local_worker.ollama_client import DEFAULT_MODEL, Generation, OllamaClient


ARMS = ("free_form", "constrained")
PROBLEM_RE = re.compile(r"(?:Problem|Observation):\s*([a-zA-Z0-9_]+)\((-?\d+),\s*(-?\d+)\)")
ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*(-?\d+)", re.IGNORECASE)


class RenderOnlyClient:
    """Deterministic client for structural evaluation without Ollama."""

    def generate_with_metrics(self, prompt: str, **kwargs: Any) -> Generation:
        system = str(kwargs.get("system", ""))
        if "choose one action" in system.lower():
            if "(none)" in prompt.split("Actions already taken this cycle:", 1)[-1]:
                response = '{"kind":"retrieve","query":"operator rule","limit":5}'
            elif "reason:" in prompt:
                match = ANSWER_RE.search(prompt)
                value = match.group(1) if match else "0"
                response = f'{{"kind":"ground","name":"answer","arguments":{{"value":{value}}}}}'
            elif "retrieve:" in prompt:
                response = '{"kind":"reason","prompt":"solve the problem"}'
            else:
                match = ANSWER_RE.search(prompt)
                value = match.group(1) if match else "0"
                response = f'{{"kind":"ground","name":"answer","arguments":{{"value":{value}}}}}'
        else:
            match = PROBLEM_RE.search(prompt)
            if match:
                name, raw_a, raw_b = match.groups()
                a, b = int(raw_a), int(raw_b)
                function, _rule = OPERATORS[name]
                response = f"ANSWER={function(a, b)}"
            else:
                response = "ANSWER=0"
        return Generation(response, 0, 0, 0, 0)


def _answer_cycle(
    controller: CoALAController,
    adapter: OllamaCoALAAdapter,
    problem: dict[str, Any],
) -> tuple[int | None, str, tuple[str, ...], bool]:
    result = controller.run_cycle(
        goal=f"answer {problem['operator']} problem",
        observation=f"{problem['operator']}({problem['a']}, {problem['b']})",
        policy=adapter.policy,
    )
    event_kinds = tuple(event.action.kind.value for event in result.events)
    retrieved_before_terminal = (
        ActionKind.RETRIEVE.value in event_kinds
        and event_kinds.index(ActionKind.RETRIEVE.value) < len(event_kinds) - 1
    )
    # The model may or may not emit a REASON step before grounding. Prefer the
    # terminal GROUND value as the answer; otherwise fall back to the reason text.
    reason_event = next(
        (event for event in result.events if event.action.kind is ActionKind.REASON), None
    )
    reason_output = reason_event.output if reason_event else ""
    if result.terminal_action.kind is ActionKind.GROUND:
        answer = parse_answer(result.output)
    else:
        answer = parse_answer(reason_output) if reason_output else None
    return answer, reason_output or result.output, event_kinds, retrieved_before_terminal


def _learn_verified_rule(controller: CoALAController, problem: dict[str, Any]) -> str:
    action = CognitiveAction.learn(
        LongTermMemoryKind.SEMANTIC,
        topic=f"operator:{problem['operator']}",
        content=f"Verified operator rule: {problem['rule']}",
        key=f"operator:{problem['operator']}",
        value=problem["rule"],
        confidence=1.0,
        utility=1.0,
        source="graded-operator-feedback",
    )
    result = controller.run_cycle(
        goal=f"learn verified rule for {problem['operator']}",
        observation=f"Feedback after grading: {problem['rule']}",
        policy=lambda _context: action,
    )
    return result.output


def evaluate(
    sequence: list[dict[str, Any]],
    *,
    model: str,
    seed: int,
    clients: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run both action-production arms over one identical problem sequence."""
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        client = (clients or {}).get(arm) or RenderOnlyClient()
        adapter = OllamaCoALAAdapter(
            client,
            model=model,
            grounding_actions=["answer"],
            max_reason_tokens=96,
            require_retrieval_before_terminal=True,
            constrained=arm == "constrained",
        )
        memory = PersistentBrainRuntime()
        controller = CoALAController(
            memory,
            reasoner=adapter.reason,
            grounding={"answer": lambda arguments, _context: str(arguments.get("value"))},
            max_internal_actions=2,
        )
        learned: set[str] = set()
        arm_rows: list[dict[str, Any]] = []
        for problem in sequence:
            # A real model can emit valid-but-non-terminating actions (or repeated
            # malformed-JSON fallbacks) that never reach a terminal within the cycle
            # budget. That is a measured failure of this problem, not a harness crash:
            # count it as incomplete/incorrect and continue. Symmetric across arms.
            try:
                answer, response, event_kinds, retrieved_before_terminal = _answer_cycle(
                    controller, adapter, problem
                )
                incomplete = False
            except CycleLimitExceeded:
                answer, response, event_kinds, retrieved_before_terminal = None, "", (), False
                incomplete = True
            correct = answer == problem["expected"]
            learned_after_feedback = problem["operator"] not in learned
            if learned_after_feedback:
                _learn_verified_rule(controller, problem)
                learned.add(problem["operator"])
            row = {
                "seed": seed,
                "arm": arm,
                **problem,
                "answer": answer,
                "response": response,
                "correct": correct,
                "incomplete": incomplete,
                "event_kinds": list(event_kinds),
                "retrieved_before_terminal": retrieved_before_terminal,
                "learned_after_feedback": learned_after_feedback,
            }
            rows.append(row)
            arm_rows.append(row)
        metrics = adapter.metrics.as_dict()
        calls = metrics["calls"]
        completed = [row for row in arm_rows if not row["incomplete"]]
        arms[arm] = {
            **metrics,
            "malformed_action_rate": metrics["malformed_actions"] / calls if calls else 0.0,
            "task_completion_accuracy": sum(row["correct"] for row in arm_rows) / len(arm_rows),
            "cycle_completion_rate": len(completed) / len(arm_rows) if arm_rows else 0.0,
            # forced_retrieval invariant is a sanity check over cycles that reached a
            # terminal; an incomplete (crashed) cycle has no terminal to precede.
            "forced_retrieval_ok": all(row["retrieved_before_terminal"] for row in completed) if completed else False,
        }

    return {
        "benchmark": "coala-constrained-action-v1",
        "model": model,
        "seed": seed,
        "arms": arms,
        "rows": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def run_ollama(*, model: str, seed: int, render_only: bool = False) -> dict[str, Any]:
    if render_only:
        clients = {arm: RenderOnlyClient() for arm in ARMS}
    else:
        client = OllamaClient(timeout_seconds=180.0)
        clients = {arm: client for arm in ARMS}
    return evaluate(build_sequence(seed), model=model, seed=seed, clients=clients)


def _default_output(model: str) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
    return f"results/brain_runtime/constrained_action_{safe_model}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default=None)
    parser.add_argument("--render-only", action="store_true", help="Use a deterministic client without Ollama.")
    args = parser.parse_args()
    payload = run_ollama(model=args.model, seed=args.seed, render_only=args.render_only)
    output = Path(args.out or _default_output(args.model))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm, metrics in payload["arms"].items():
        print(
            f"{arm:<12} malformed={metrics['malformed_action_rate']:.3f} "
            f"accuracy={metrics['task_completion_accuracy']:.3f} "
            f"forced_retrieval_ok={metrics['forced_retrieval_ok']}"
        )


if __name__ == "__main__":
    main()
