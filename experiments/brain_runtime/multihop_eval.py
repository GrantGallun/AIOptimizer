#!/usr/bin/env python3
"""Prereg v6: multi-hop composition — does forced structure help THINKING, not just format?

Every prior task was single-hop ("apply one retrieved rule"), and results saturated at 0/1,
so gates measured cliffs. Here each problem composes two novel operators —
``outer(inner(a, b), c)`` — so the model must retrieve BOTH rules and chain them. Two arms
from the integrated-kernel eval, the two that still matter after HYP-27/29:

- ``prompted``: encoder retrieval + strong prompt, NO deterministic invariants.
- ``full_kernel``: encoder retrieval + constrained actions + ordered invariants
  (retrieve-before-terminal, reason-after-last-retrieve).

A problem counts as recurrence only when BOTH operators were already learned. Answers are
graded from the last REASON output; incomplete cycles count as wrong. Fable reads the gate
(fresh hidden seeds from stats.FRESH_HIDDEN_SEEDS['v6']; 101/103/107 are burned).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.coala import ActionKind, CoALAController, CycleLimitExceeded
from experiments.brain_runtime.coala_learning_eval import parse_answer
from experiments.brain_runtime.constrained_action_eval import _learn_verified_rule
from experiments.brain_runtime.context_selection import make_operators
from experiments.brain_runtime.integrated_kernel_eval import ScaleMockClient, _make_adapter, _memory
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient

ARMS = ("prompted", "full_kernel")
DEFAULT_SEED = 20260711


def build_composed_sequence(
    operators: list[dict[str, Any]], *, seed: int, n_problems: int = 60
) -> list[dict[str, Any]]:
    """Depth-2 composition problems; recurrence requires BOTH operators learned."""
    if len(operators) < 2:
        raise ValueError("need at least two operators to compose")
    rng = random.Random(seed)
    by_name = {op["name"]: op for op in operators}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for step in range(n_problems):
        outer, inner = rng.sample(list(by_name), 2)
        a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
        inner_value = by_name[inner]["fn"](a, b)
        rows.append(
            {
                "step": step,
                "operator": f"{outer}∘{inner}",
                "outer": outer,
                "inner": inner,
                "a": a,
                "b": b,
                "c": c,
                "inner_value": inner_value,
                "expected": int(by_name[outer]["fn"](inner_value, c)),
                "rules": {
                    inner: str(by_name[inner]["rule_text"]),
                    outer: str(by_name[outer]["rule_text"]),
                },
                "first_appearance": not {outer, inner} <= seen,
            }
        )
        seen.update({outer, inner})
    return rows


def _observation(problem: dict[str, Any]) -> str:
    return f"{problem['outer']}({problem['inner']}({problem['a']}, {problem['b']}), {problem['c']})"


def _answer_cycle(controller: CoALAController, adapter: Any, problem: dict[str, Any]):
    result = controller.run_cycle(
        goal=f"answer the composed {problem['outer']} of {problem['inner']} problem",
        observation=_observation(problem),
        policy=adapter.policy,
    )
    event_kinds = tuple(event.action.kind.value for event in result.events)
    reason_events = [event for event in result.events if event.action.kind is ActionKind.REASON]
    reason_output = reason_events[-1].output if reason_events else ""
    return (parse_answer(reason_output) if reason_output else None), reason_output, event_kinds


def evaluate(operators, sequence, *, model: str, seed: int, clients: dict[str, Any] | None = None,
             arms_to_run: tuple[str, ...] = ARMS, max_internal_actions: int = 4):
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    arms: dict[str, dict[str, Any]] = {}
    for arm in arms_to_run:
        client = (clients or {}).get(arm) or ScaleMockClient()
        # v6.1: two-hop arithmetic with shown work truncates at the single-hop budget of 96
        # (dev diagnosis: correct reasoning cut mid-second-hop, last-integer grading then fails).
        adapter = _make_adapter(client, model, arm, max_reason_tokens=256)
        controller = CoALAController(
            _memory(arm),
            reasoner=adapter.reason,
            grounding={"answer": lambda arguments, _context: str(arguments.get("value"))},
            max_internal_actions=max_internal_actions,
        )
        learned: set[str] = set()
        arm_rows: list[dict[str, Any]] = []
        for problem in sequence:
            try:
                answer, response, event_kinds = _answer_cycle(controller, adapter, problem)
                incomplete = False
            except (CycleLimitExceeded, PermissionError):
                answer, response, event_kinds, incomplete = None, "", (), True
            for name, rule in problem["rules"].items():
                if name not in learned:
                    _learn_verified_rule(controller, {"operator": name, "rule": rule})
                    learned.add(name)
            arm_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    **{k: v for k, v in problem.items() if k != "rules"},
                    "answer": answer,
                    "response": response,
                    "correct": answer == problem["expected"],
                    "incomplete": incomplete,
                    "event_kinds": list(event_kinds),
                }
            )
        rows.extend(arm_rows)
        recurrence = [r for r in arm_rows if not r["first_appearance"]]
        completed = [r for r in arm_rows if not r["incomplete"]]
        metrics = adapter.metrics.as_dict()
        acc = lambda g: (sum(r["correct"] for r in g) / len(g)) if g else 0.0
        arms[arm] = {
            "arm": arm,
            **metrics,
            "recurrence_accuracy": acc(recurrence),
            "recurrence_tasks": len(recurrence),
            "first_appearance_accuracy": acc([r for r in arm_rows if r["first_appearance"]]),
            "cycle_completion_rate": len(completed) / len(arm_rows) if arm_rows else 0.0,
            "malformed_action_rate": metrics["malformed_actions"] / metrics["calls"] if metrics["calls"] else 0.0,
        }
    return {
        "benchmark": "multihop-composition-v6",
        "model": model,
        "seed": seed,
        "max_internal_actions": max_internal_actions,
        "operator_count": len(operators),
        "rows": rows,
        "arms": arms,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def run_ollama(*, model: str, seed: int, operator_count: int = 12, n_problems: int = 60,
               render_only: bool = False, arms_to_run: tuple[str, ...] = ARMS) -> dict[str, Any]:
    operators = make_operators(operator_count, seed=seed)
    sequence = build_composed_sequence(operators, seed=seed, n_problems=n_problems)
    if render_only:
        clients = {arm: ScaleMockClient() for arm in arms_to_run}
    else:
        client = OllamaClient(timeout_seconds=180.0)
        clients = {arm: client for arm in arms_to_run}
    return evaluate(operators, sequence, model=model, seed=seed, clients=clients, arms_to_run=arms_to_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-operators", type=int, default=12)
    parser.add_argument("--n-problems", type=int, default=60)
    parser.add_argument("--out", default=None)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    payload = run_ollama(model=args.model, seed=args.seed, operator_count=args.n_operators,
                         n_problems=args.n_problems, render_only=args.render_only)
    out = args.out or f"results/brain_runtime/multihop_{re.sub(r'[^A-Za-z0-9]+','_',args.model)}_{args.seed}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm, s in payload["arms"].items():
        print(f"{arm:<12} recurrence={s['recurrence_accuracy']:.3f} completion={s['cycle_completion_rate']:.3f} malformed={s['malformed_action_rate']:.3f}")


if __name__ == "__main__":
    main()
