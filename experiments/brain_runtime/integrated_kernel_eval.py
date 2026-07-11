#!/usr/bin/env python3
"""Prereg v4 capstone: the full deterministic kernel vs the naive LLM-glue stack, at scale.

Both arms let the MODEL choose its own CoALA actions as JSON (so action validity is measured,
not assumed) over the scaled novel-operator learning task (30 operators x 5 repetitions):

- ``naive``: jaccard retrieval (PersistentBrainRuntime), free-form actions, NO structural
  invariants. How an agent looks when the LLM decides everything.
- ``full_kernel``: encoder top-k retrieval (EncoderMemoryRuntime), constrained decoding with the
  per-kind conditional schema (HYP-25), and both invariants (retrieval-before-terminal +
  reason-before-ground).

Answers are graded from the REASON output (HYP-25). Non-terminating cycles are counted incomplete
(incorrect), never crashes. Fable reads the frozen hidden gate and writes the verdict; this module
computes components only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.coala import ActionKind, CoALAController, CycleLimitExceeded
from experiments.brain_runtime.coala_learning_eval import parse_answer
from experiments.brain_runtime.coala_learning_eval_scale import (
    DEFAULT_OPERATOR_COUNT,
    DEFAULT_REPETITIONS,
    DEFAULT_SEED,
    EncoderMemoryRuntime,
    build_sequence,
)
from experiments.brain_runtime.coala_ollama import ACTION_SCHEMA_CONDITIONAL, OllamaCoALAAdapter
from experiments.brain_runtime.constrained_action_eval import _learn_verified_rule
from experiments.brain_runtime.context_selection import make_operators
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime
from experiments.local_worker.ollama_client import DEFAULT_MODEL, Generation, OllamaClient

ARMS = ("naive", "full_kernel")


class ScaleMockClient:
    """Deterministic, operator-agnostic client for model-free structural tests.

    Drives every cycle retrieve -> reason -> ground with valid action JSON and a
    fixed ANSWER (render-only exercises plumbing/metrics, not answer correctness,
    since the scaled task uses procedurally generated operators).
    """

    def generate_with_metrics(self, prompt: str, **kwargs: Any) -> Generation:
        system = str(kwargs.get("system", "")).lower()
        if "choose one action" not in system:  # REASON_SYSTEM -> produce an answer
            return Generation("ANSWER=0", 0, 0, 0, 0)
        taken = prompt.split("Actions already taken this cycle:", 1)[-1]
        if "retrieve" not in taken:
            response = '{"kind":"retrieve","query":"operator rule","limit":5}'
        elif "reason" not in taken:
            response = '{"kind":"reason","prompt":"solve the problem"}'
        else:
            response = '{"kind":"ground","name":"answer","arguments":{"value":0}}'
        return Generation(response, 0, 0, 0, 0)


def _make_adapter(client: Any, model: str, arm: str) -> OllamaCoALAAdapter:
    if arm == "full_kernel":
        return OllamaCoALAAdapter(
            client,
            model=model,
            grounding_actions=["answer"],
            max_reason_tokens=96,
            require_retrieval_before_terminal=True,
            require_reason_before_ground=True,
            constrained=True,
            action_schema=ACTION_SCHEMA_CONDITIONAL,
        )
    return OllamaCoALAAdapter(  # naive: pure LLM-glue
        client,
        model=model,
        grounding_actions=["answer"],
        max_reason_tokens=96,
        require_retrieval_before_terminal=False,
        require_reason_before_ground=False,
        constrained=False,
    )


def _memory(arm: str) -> PersistentBrainRuntime:
    return EncoderMemoryRuntime() if arm == "full_kernel" else PersistentBrainRuntime()


def _answer_cycle(controller: CoALAController, adapter: OllamaCoALAAdapter, problem: dict[str, Any]):
    result = controller.run_cycle(
        goal=f"answer {problem['operator']} problem",
        observation=f"{problem['operator']}({problem['a']}, {problem['b']})",
        policy=adapter.policy,
    )
    event_kinds = tuple(event.action.kind.value for event in result.events)
    reason_events = [event for event in result.events if event.action.kind is ActionKind.REASON]
    reason_output = reason_events[-1].output if reason_events else ""
    answer = parse_answer(reason_output) if reason_output else None
    return answer, reason_output, event_kinds


def evaluate(operators, sequence, *, model: str, seed: int, clients: dict[str, Any] | None = None):
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        client = (clients or {}).get(arm) or ScaleMockClient()
        adapter = _make_adapter(client, model, arm)
        memory = _memory(arm)
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
                answer, response, event_kinds = _answer_cycle(controller, adapter, problem)
                incomplete = False
            except CycleLimitExceeded:
                answer, response, event_kinds, incomplete = None, "", (), True
            correct = answer == problem["expected"]
            if problem["operator"] not in learned:
                _learn_verified_rule(controller, problem)
                learned.add(problem["operator"])
            arm_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    **problem,
                    "answer": answer,
                    "response": response,
                    "correct": correct,
                    "incomplete": incomplete,
                    "event_kinds": list(event_kinds),
                }
            )
        rows.extend(arm_rows)
        arms[arm] = _summarize(arm, arm_rows, adapter.metrics.as_dict())
    gap = arms["full_kernel"]["recurrence_accuracy"] - arms["naive"]["recurrence_accuracy"]
    return {
        "benchmark": "integrated-kernel-v4",
        "model": model,
        "seed": seed,
        "operator_count": len(operators),
        "repetitions": len(sequence) // len(operators),
        "rows": rows,
        "arms": arms,
        "gate": {
            "owner": "fable",
            "required_recurrence_gap": 0.30,
            "observed_recurrence_gap": gap,
            "full_kernel_malformed_rate": arms["full_kernel"]["malformed_action_rate"],
            "naive_malformed_rate": arms["naive"]["malformed_action_rate"],
            "note": "Fable reads the hidden gate and writes the verdict.",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _summarize(arm: str, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    first = [r for r in rows if r["first_appearance"]]
    recurrence = [r for r in rows if not r["first_appearance"]]
    completed = [r for r in rows if not r["incomplete"]]
    calls = metrics["calls"]
    acc = lambda group: (sum(r["correct"] for r in group) / len(group)) if group else 0.0
    # Structural participation: did the cycle actually retrieve / reason before ending?
    # This is where the naive stack fails (it grounds immediately) — the real separator.
    retrieved = [r for r in rows if "retrieve" in r["event_kinds"]]
    reasoned = [r for r in rows if "reason" in r["event_kinds"]]
    return {
        "arm": arm,
        **metrics,
        "first_appearance_accuracy": acc(first),
        "recurrence_accuracy": acc(recurrence),
        "recurrence_tasks": len(recurrence),
        "malformed_action_rate": metrics["malformed_actions"] / calls if calls else 0.0,
        "cycle_completion_rate": len(completed) / len(rows) if rows else 0.0,
        "retrieval_participation_rate": len(retrieved) / len(rows) if rows else 0.0,
        "reasoning_participation_rate": len(reasoned) / len(rows) if rows else 0.0,
    }


def run_ollama(*, model: str, seed: int, operator_count=DEFAULT_OPERATOR_COUNT,
               repetitions=DEFAULT_REPETITIONS, render_only: bool = False) -> dict[str, Any]:
    operators = make_operators(operator_count, seed=seed)
    sequence = build_sequence(operators, seed=seed, repetitions=repetitions)
    if render_only:
        clients = {arm: ScaleMockClient() for arm in ARMS}
    else:
        client = OllamaClient(timeout_seconds=180.0)
        clients = {arm: client for arm in ARMS}
    return evaluate(operators, sequence, model=model, seed=seed, clients=clients)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-operators", type=int, default=DEFAULT_OPERATOR_COUNT)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--out", default="results/brain_runtime/integrated_kernel.json")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    payload = run_ollama(model=args.model, seed=args.seed, operator_count=args.n_operators,
                         repetitions=args.repetitions, render_only=args.render_only)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm in ARMS:
        s = payload["arms"][arm]
        print(f"{arm:<12} recurrence={s['recurrence_accuracy']:.3f} first={s['first_appearance_accuracy']:.3f} "
              f"malformed={s['malformed_action_rate']:.3f} completion={s['cycle_completion_rate']:.3f}")
    print(f"recurrence gap (full_kernel - naive) = {payload['gate']['observed_recurrence_gap']:.3f}")


if __name__ == "__main__":
    main()
