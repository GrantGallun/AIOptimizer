#!/usr/bin/env python3
"""Prereg v3.1: constrained CoALA action-validity with the fixes HYP-24 demanded.

Two changes from ``constrained_action_eval`` (v3), both frozen in PREREGISTRATION_v3.md
Amendment v3.1 BEFORE this was run:

1. The constrained arm uses ``ACTION_SCHEMA_CONDITIONAL`` — a per-kind ``oneOf`` whose
   branches mirror exactly what ``parse_action`` requires/forbids — so a schema-valid
   object is also parse-valid. (v3's permissive union only *required* ``kind``, so it
   could not guarantee malformed=0; HYP-24.)
2. Both arms enforce ``require_reason_before_ground`` and the answer is graded from the
   REASON step's output. (v3 let models ground a null value by skipping reasoning, which
   made task-completion degenerate at 0 everywhere.)

The reason-before-ground invariant is a deterministic structural mechanism applied
symmetrically to both arms, so it cannot bias the free_form-vs-constrained comparison.
Fable runs this and reads the frozen hidden gate; do not edit the gate after seeing results.
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

from experiments.brain_runtime.coala import ActionKind, CoALAController, CycleLimitExceeded
from experiments.brain_runtime.coala_learning_eval import DEFAULT_SEED, build_sequence, parse_answer
from experiments.brain_runtime.coala_ollama import ACTION_SCHEMA_CONDITIONAL, OllamaCoALAAdapter
from experiments.brain_runtime.constrained_action_eval import RenderOnlyClient, _learn_verified_rule
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient

ARMS = ("free_form", "constrained")


def _answer_cycle(controller: CoALAController, adapter: OllamaCoALAAdapter, problem: dict[str, Any]):
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
    # v3.1 grades the answer from the model's reasoning, not the (often null) ground value.
    # Use the LAST reason event: with the retrieve-then-reason invariants it is the one
    # produced after the rule was retrieved.
    reason_events = [event for event in result.events if event.action.kind is ActionKind.REASON]
    reason_output = reason_events[-1].output if reason_events else ""
    answer = parse_answer(reason_output) if reason_output else None
    return answer, reason_output, event_kinds, retrieved_before_terminal


def evaluate(sequence: list[dict[str, Any]], *, model: str, seed: int, clients: dict[str, Any] | None = None):
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
            require_reason_before_ground=True,
            constrained=arm == "constrained",
            action_schema=ACTION_SCHEMA_CONDITIONAL,
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
            try:
                answer, response, event_kinds, retrieved_before_terminal = _answer_cycle(
                    controller, adapter, problem
                )
                incomplete = False
            except (CycleLimitExceeded, PermissionError):
                answer, response, event_kinds, retrieved_before_terminal = None, "", (), False
                incomplete = True
            correct = answer == problem["expected"]
            if problem["operator"] not in learned:
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
            "forced_retrieval_ok": all(row["retrieved_before_terminal"] for row in completed) if completed else False,
        }
    return {
        "benchmark": "coala-constrained-action-v3.1",
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
    return f"results/brain_runtime/constrained_action_v31_{safe_model}.json"


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
            f"completion={metrics['cycle_completion_rate']:.3f} "
            f"forced_retrieval_ok={metrics['forced_retrieval_ok']}"
        )


if __name__ == "__main__":
    main()
