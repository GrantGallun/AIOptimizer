#!/usr/bin/env python3
"""Prereg v5: the idiomatic LangGraph agent vs the AIOptimizer deterministic kernel.

The honest product comparison. We build the novel-operator learning agent the canonical
LangGraph way — a StateGraph whose LLM "router" node DECIDES its own next step (retrieve the
rule, or answer) via conditional edges — and compare it to ``full_kernel`` (prereg v4), which
FORCES retrieve->reason->ground. Retrieval quality is held equal (both use the same encoder
top-k over the same learned-rule store); the ONLY difference is control-flow determinism:
LangGraph lets the model choose whether to retrieve; the kernel guarantees it.

This is not "LangGraph the library is bad" — the kernel's mechanisms could be added to a LangGraph
graph. It tests whether the *idiomatic LLM-driven control flow* (how agents are usually built)
hits the same structural failure our naive arm did (HYP-26): answering without retrieving.

Run: ``python experiments/brain_runtime/langgraph_agent_eval.py --model qwen3:8b --seed 101``
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, StateGraph

from experiments.brain_runtime.coala_learning_eval import parse_answer
from experiments.brain_runtime.coala_learning_eval_scale import DEFAULT_SEED, build_sequence
from experiments.brain_runtime.context_selection import make_operators, select_topk
from experiments.local_worker.ollama_client import DEFAULT_MODEL, Generation, OllamaClient

MAX_ROUTER_STEPS = 3

ROUTER_SYSTEM = (
    "You are the controller of a problem-solving agent. Decide the single next step. "
    'Return exactly one JSON object: {"action":"retrieve"} to look up the operator\'s rule from '
    'memory, or {"action":"answer"} to give the final answer. Retrieve before answering if you do '
    "not already have the rule."
)
COMPUTE_SYSTEM = (
    "Compute the operator result. Use the authorized rule if one is shown; do not invent a rule. "
    "Show the arithmetic and end with exactly ANSWER=<integer>."
)
_ACTION_RE = re.compile(r'"action"\s*:\s*"(retrieve|answer)"', re.IGNORECASE)


class AgentState(TypedDict):
    operator: str
    a: int
    b: int
    retrieved_rule: Optional[str]
    action: Optional[str]
    steps: int
    answer: Optional[int]
    retrieved: bool


def _encoder_retrieve(operator: str, store: dict[str, str], k: int = 1) -> Optional[str]:
    """Encoder top-k over the learned-rule store, newline-joined for the answer prompt.

    k=1 was the original HYP-28 run; the kernel retrieves limit=5 and renders every
    candidate, so a k-matched comparison (HYP-31) must pass k>=3 here.
    """
    if not store:
        return None
    rules = list(store.values())
    if len(rules) <= k:
        return "\n".join(rules)
    return "\n".join(select_topk(f"{operator} rule", rules, k))


class LangGraphAgent:
    """Idiomatic LLM-routed agent; ``store`` is the shared learned-rule memory."""

    def __init__(self, client: Any, model: str, store: dict[str, str], retrieve_k: int = 1) -> None:
        self.client = client
        self.model = model
        self.store = store
        self.retrieve_k = retrieve_k
        self.app = self._build()

    def _router(self, state: AgentState) -> dict[str, Any]:
        have = state.get("retrieved_rule")
        prompt = (
            f"Problem: {state['operator']}({state['a']}, {state['b']}).\n"
            f"Rule retrieved so far: {have if have else '(none)'}\n"
            "Choose the next action."
        )
        text = self.client.generate_with_metrics(
            prompt, model=self.model, system=ROUTER_SYSTEM, temperature=0.0, max_tokens=24
        ).text
        match = _ACTION_RE.search(text)
        action = match.group(1).lower() if match else "answer"
        return {"action": action, "steps": state["steps"] + 1}

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        rule = _encoder_retrieve(state["operator"], self.store, self.retrieve_k)
        return {"retrieved_rule": rule, "retrieved": True}

    def _answer(self, state: AgentState) -> dict[str, Any]:
        rule = state.get("retrieved_rule")
        prompt = (
            f"{'Authorized rules (use the one for this operator):' + chr(10) + rule if rule else 'No rule was retrieved.'}\n"
            f"Compute {state['operator']}({state['a']}, {state['b']})."
        )
        text = self.client.generate_with_metrics(
            prompt, model=self.model, system=COMPUTE_SYSTEM, temperature=0.0, max_tokens=96
        ).text
        return {"answer": parse_answer(text)}

    def _route(self, state: AgentState) -> str:
        if state["steps"] > MAX_ROUTER_STEPS:
            return "answer"
        return "retrieve" if state.get("action") == "retrieve" else "answer"

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.set_entry_point("router")
        graph.add_conditional_edges("router", self._route, {"retrieve": "retrieve", "answer": "answer"})
        graph.add_edge("retrieve", "router")  # loop back so the model may answer after retrieving
        graph.add_edge("answer", END)
        return graph.compile()

    def solve(self, operator: str, a: int, b: int) -> tuple[Optional[int], bool]:
        final = self.app.invoke(
            {"operator": operator, "a": a, "b": b, "retrieved_rule": None,
             "action": None, "steps": 0, "answer": None, "retrieved": False}
        )
        return final.get("answer"), bool(final.get("retrieved"))


class RouterMockClient:
    """Model-free client: router always retrieves once then answers; answer returns ANSWER=0."""

    def generate_with_metrics(self, prompt: str, **kwargs: Any) -> Generation:
        system = str(kwargs.get("system", "")).lower()
        if "controller" in system:  # ROUTER_SYSTEM
            action = "retrieve" if "(none)" in prompt else "answer"
            return Generation(f'{{"action":"{action}"}}', 0, 0, 0, 0)
        return Generation("ANSWER=0", 0, 0, 0, 0)


def evaluate(operators, sequence, *, model: str, seed: int, client: Any, retrieve_k: int = 1) -> dict[str, Any]:
    started = time.perf_counter()
    store: dict[str, str] = {}
    agent = LangGraphAgent(client, model, store, retrieve_k=retrieve_k)
    learned: set[str] = set()
    rows: list[dict[str, Any]] = []
    for problem in sequence:
        answer, retrieved = agent.solve(problem["operator"], problem["a"], problem["b"])
        rows.append({
            "seed": seed, **problem, "answer": answer,
            "correct": answer == problem["expected"], "retrieved": retrieved,
        })
        if problem["operator"] not in learned:  # same learning protocol as the kernel
            store[problem["operator"]] = str(problem["rule"])
            learned.add(problem["operator"])
    recurrence = [r for r in rows if not r["first_appearance"]]
    first = [r for r in rows if r["first_appearance"]]
    acc = lambda g: (sum(r["correct"] for r in g) / len(g)) if g else 0.0
    return {
        "benchmark": "langgraph-idiomatic-agent-v5",
        "model": model,
        "seed": seed,
        "operator_count": len(operators),
        "rows": rows,
        "retrieve_k": agent.retrieve_k,
        "summary": {
            "arm": "langgraph_idiomatic",
            "recurrence_accuracy": acc(recurrence),
            "first_appearance_accuracy": acc(first),
            "recurrence_tasks": len(recurrence),
            "retrieval_participation_rate": sum(r["retrieved"] for r in rows) / len(rows) if rows else 0.0,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def run(*, model: str, seed: int, operator_count: int = 30, repetitions: int = 5,
        render_only: bool = False, retrieve_k: int = 1) -> dict[str, Any]:
    operators = make_operators(operator_count, seed=seed)
    sequence = build_sequence(operators, seed=seed, repetitions=repetitions)
    client = RouterMockClient() if render_only else OllamaClient(timeout_seconds=180.0)
    return evaluate(operators, sequence, model=model, seed=seed, client=client, retrieve_k=retrieve_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-operators", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--retrieve-k", type=int, default=1)
    args = parser.parse_args()
    payload = run(model=args.model, seed=args.seed, operator_count=args.n_operators,
                  repetitions=args.repetitions, render_only=args.render_only, retrieve_k=args.retrieve_k)
    out = args.out or f"results/brain_runtime/langgraph_agent_{re.sub(r'[^A-Za-z0-9]+','_',args.model)}_{args.seed}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s = payload["summary"]
    print(f"langgraph_idiomatic recurrence={s['recurrence_accuracy']:.3f} "
          f"first={s['first_appearance_accuracy']:.3f} "
          f"retrieval_participation={s['retrieval_participation_rate']:.3f}")


if __name__ == "__main__":
    main()
