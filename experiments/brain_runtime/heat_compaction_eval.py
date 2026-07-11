#!/usr/bin/env python3
"""Prereg v12: ACT-R observed-usage activation vs recency at compaction time.

Four phases per arm (phases 1–3 are model-free; only the test phase calls the model):
1. LEARN 30 verified operator rules into ``PersistentBrainRuntime``.
2. USAGE: a seeded retrieval sequence exercises a HOT subset (8 rules × 4), incrementing the
   runtime's real ``uses``/``last_accessed`` counters. Cold rules get 0 touches.
3. COMPACTION: the long-term store is reduced to K=12 by the arm's policy
   (``compaction_policies``: newest / random / actr — frozen formulas, t0034).
4. TEST: 24 problems (16 hot / 8 cold; frozen mix unknown at compaction time); a hand-coded
   retrieve→reason cycle measures whether the needed rule SURVIVED and the model applies it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.compaction_policies import keep_actr, keep_newest, keep_random
from experiments.brain_runtime.context_selection import make_operators, parse_int
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient

ARMS = {"newest": keep_newest, "random_k": keep_random, "actr": keep_actr}
KEEP = 12
N_HOT, HOT_TOUCHES = 8, 4
N_TEST_HOT, N_TEST_COLD = 16, 8
DEFAULT_SEED = 20260711


def _build_runtime(operators: list[dict[str, Any]], seed: int) -> tuple[PersistentBrainRuntime, list[str]]:
    """Phases 1–2: learn every rule, then exercise the hot subset (model-free)."""
    runtime = PersistentBrainRuntime()
    for op in operators:
        runtime.remember(
            topic=f"operator:{op['name']}",
            content=f"Verified operator rule: {op['rule_text']}",
            kind="semantic",
            key=f"operator:{op['name']}",
            value=str(op["rule_text"]),
            confidence=1.0,
            utility=1.0,
            source="graded-operator-feedback",
            long_term=True,
        )
    rng = random.Random(seed)
    hot = [op["name"] for op in rng.sample(operators, N_HOT)]
    touch_order = hot * HOT_TOUCHES
    rng.shuffle(touch_order)
    # Usage = verified APPLICATION events, counted directly. Retrieval-based counting was
    # tried first and mis-attributes usage: jaccard surfaces the same 3 items for every
    # "operator:<name> rule" query (shared tokens only) and the runtime's use_bonus then
    # amplifies the winners — a rich-get-richer loop (substrate finding, see prereg note).
    by_key = {item.key: item for item in runtime.long_term_memory.values()}
    for name in touch_order:
        runtime.tick()
        item = by_key[f"operator:{name}"]
        item.uses += 1
        item.last_accessed = runtime.clock
    return runtime, hot


def _compact(runtime: PersistentBrainRuntime, policy, seed: int) -> None:
    items = list(runtime.long_term_memory.values())
    # Decorrelated seed: the hot-subset sampler and keep_random otherwise consume the same
    # Mersenne stream over same-ordered lists, making the "random" control track the
    # treatment assignment (caught at render level: random kept exactly the 8 hot rules).
    kept = policy(items, KEEP, clock=runtime.clock, seed=seed + 104729)
    runtime.long_term_memory = {item.id: item for item in kept}


def _test_problems(operators, hot: list[str], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 1)
    by_name = {op["name"]: op for op in operators}
    cold = [n for n in by_name if n not in hot]
    picks = [rng.choice(hot) for _ in range(N_TEST_HOT)] + rng.sample(cold, N_TEST_COLD)
    rng.shuffle(picks)
    problems = []
    for name in picks:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        problems.append({"operator": name, "a": a, "b": b,
                         "expected": by_name[name]["fn"](a, b), "hot": name in hot})
    return problems


def _answer(client: OllamaClient, model: str, runtime: PersistentBrainRuntime, problem) -> int | None:
    retrieved = runtime.retrieve(f"operator:{problem['operator']} rule", limit=3)
    rules = [str(item.value) for item in retrieved
             if item.key == f"operator:{problem['operator']}" and item.value]
    if not rules:
        return None  # the rule did not survive compaction — unanswerable
    prompt = ("Apply the rule exactly and compute the result. Reply with ONLY an integer.\n"
              f"Rule: {rules[0]}\n{problem['operator']}({problem['a']}, {problem['b']}) = ?")
    return parse_int(client.generate_with_metrics(prompt, model=model, max_tokens=24).text)


def run(*, model: str, seed: int, endpoint: str | None = None, render_only: bool = False):
    started = time.perf_counter()
    operators = make_operators(30, seed=seed)
    client = None if render_only else OllamaClient(endpoint=endpoint, timeout_seconds=240.0)
    rows, summaries = [], {}
    for arm, policy in ARMS.items():
        runtime, hot = _build_runtime(operators, seed)   # identical pre-compaction state per arm
        _compact(runtime, policy, seed)
        surviving_keys = {item.key for item in runtime.long_term_memory.values()}
        problems = _test_problems(operators, hot, seed)
        arm_rows = []
        for problem in problems:
            survived = f"operator:{problem['operator']}" in surviving_keys
            answer = None
            if not render_only and survived:
                answer = _answer(client, model, runtime, problem)
            correct = answer == problem["expected"]
            arm_rows.append({"arm": arm, **problem, "rule_survived": survived,
                             "answer": answer, "correct": correct})
        rows.extend(arm_rows)
        hot_rows = [r for r in arm_rows if r["hot"]]
        cold_rows = [r for r in arm_rows if not r["hot"]]
        summaries[arm] = {
            "arm": arm,
            "hot_survival": sum(r["rule_survived"] for r in hot_rows) / len(hot_rows),
            "cold_survival": sum(r["rule_survived"] for r in cold_rows) / len(cold_rows),
            **({} if render_only else {
                "accuracy": sum(r["correct"] for r in arm_rows) / len(arm_rows),
                "hot_accuracy": sum(r["correct"] for r in hot_rows) / len(hot_rows),
                "cold_accuracy": sum(r["correct"] for r in cold_rows) / len(cold_rows),
            }),
        }
    return {"benchmark": "heat-compaction-v12", "model": model, "seed": seed, "keep": KEEP,
            "rows": rows, "summaries": summaries,
            "elapsed_seconds": round(time.perf_counter() - started, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    payload = run(model=args.model, seed=args.seed, endpoint=args.endpoint,
                  render_only=args.render_only)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm, s in payload["summaries"].items():
        extras = f" acc={s['accuracy']:.2f} (hot {s['hot_accuracy']:.2f} / cold {s['cold_accuracy']:.2f})" if "accuracy" in s else ""
        print(f"{arm:<10} hot_survival={s['hot_survival']:.2f} cold_survival={s['cold_survival']:.2f}{extras}")


if __name__ == "__main__":
    main()
