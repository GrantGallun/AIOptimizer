#!/usr/bin/env python3
"""Context selection vs dumping at scale ("lost in the middle").

HYPOTHESIS (Fable-owned): dumping all rules into context works when context is
small, but degrades as context grows (lost-in-the-middle); encoder-scored
top-k selection holds accuracy at large context. So dump beats selection at
small N, selection beats dump at large N.

Each "operator" is a synthetic function with a plain-English rule (e.g.
"glorp(a,b) = a + b + 3"). A problem asks the model to compute one target
operator's result given a haystack of N candidate rules (the target's rule
plus N-1 unrelated distractor rules). Three arms share the same problem:

- dump_all: every candidate rule is placed in the prompt (naive context stuffing).
- selected: only the top-k rules by embedding similarity to the operator name
  are kept (reuses `EncoderLeakJudge._embed`, the same all-MiniLM encoder used
  for leak detection elsewhere in brain runtime).
- oracle: only the target's own rule is given (upper bound / sanity check).
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

from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient


_CONSONANTS = "bcdfgklmnprstvz"
_VOWELS = "aeiou"

_INT_RE = re.compile(r"-?\d+")

CONTEXT_SIZES: tuple[int, ...] = (5, 15, 30)


def _make_name(rng: random.Random) -> str:
    """A novel 5-letter consonant-vowel-consonant-vowel-consonant word, e.g. 'glorp'."""
    letters = [
        rng.choice(_CONSONANTS),
        rng.choice(_CONSONANTS),
        rng.choice(_VOWELS),
        rng.choice(_CONSONANTS),
        rng.choice(_VOWELS),
    ]
    return "".join(letters)


def _make_template(rng: random.Random, name: str) -> tuple[str, Callable[[int, int], int]]:
    template = rng.choice(["add_k", "mul_m_add", "abs_diff_k", "max_mul_m"])
    if template == "add_k":
        k = rng.randint(1, 9)
        rule_text = f"{name}(a,b) = a + b + {k}"
        fn = lambda a, b, k=k: a + b + k
    elif template == "mul_m_add":
        m = rng.randint(2, 4)
        rule_text = f"{name}(a,b) = (a times {m}) + b"
        fn = lambda a, b, m=m: (a * m) + b
    elif template == "abs_diff_k":
        k = rng.randint(1, 9)
        rule_text = f"{name}(a,b) = |a - b| + {k}"
        fn = lambda a, b, k=k: abs(a - b) + k
    else:  # max_mul_m
        m = rng.randint(2, 4)
        rule_text = f"{name}(a,b) = {m} times max(a, b)"
        fn = lambda a, b, m=m: m * max(a, b)
    return rule_text, fn


def make_operators(count: int = 30, seed: int = 0) -> list[dict[str, Any]]:
    """Generate `count` synthetic operators with unique names and a rule + callable each."""
    rng = random.Random(seed)
    names: set[str] = set()
    operators: list[dict[str, Any]] = []
    while len(operators) < count:
        name = _make_name(rng)
        if name in names:
            continue
        names.add(name)
        rule_text, fn = _make_template(rng, name)
        operators.append({"name": name, "rule_text": rule_text, "fn": fn})
    return operators


def parse_int(text: str) -> int | None:
    """First (optionally negative) integer literal in `text`, or None."""
    match = _INT_RE.search(text)
    if match is None:
        return None
    return int(match.group(0))


_judge = None


def _embed(texts: list[str]) -> Any:
    global _judge
    from experiments.brain_runtime.leak_judge import EncoderLeakJudge

    if _judge is None:
        _judge = EncoderLeakJudge()
    return _judge._embed(texts)


def select_topk(query: str, rules: list[str], k: int) -> list[str]:
    """The k rules most similar to `query` by cosine similarity, ranked descending.

    If there are k or fewer rules, returns them unchanged (no ranking needed).
    """
    if len(rules) <= k:
        return list(rules)
    vectors = _embed([query] + list(rules))
    query_vec = vectors[0]
    rule_vecs = vectors[1:]
    sims = rule_vecs @ query_vec
    ranked = sorted(range(len(rules)), key=lambda i: float(sims[i]), reverse=True)
    top_indices = ranked[:k]
    return [rules[i] for i in top_indices]


def build_prompt(op_name: str, a: int, b: int, rules: list[str]) -> str:
    lines = "\n".join(f"- {rule}" for rule in rules)
    return (
        "Compute the result. Reply with ONLY an integer, nothing else.\n\n"
        "Rules you must apply exactly:\n"
        f"{lines}\n\n"
        f"{op_name}({a}, {b}) = ?"
    )


ARMS: tuple[str, ...] = ("dump_all", "selected", "random_k", "oracle")


def run(
    client: OllamaClient | None,
    *,
    model: str = DEFAULT_MODEL,
    operators: list[dict[str, Any]],
    n_problems: int = 20,
    context_sizes: tuple[int, ...] = CONTEXT_SIZES,
    k: int = 3,
    seed: int = 0,
    render_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for problem_i in range(n_problems):
        target = rng.choice(operators)
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        expected = target["fn"](a, b)

        for n in context_sizes:
            distractor_pool = [op for op in operators if op["name"] != target["name"]]
            distractors = rng.sample(distractor_pool, min(n - 1, len(distractor_pool)))
            candidates = [target["rule_text"]] + [op["rule_text"] for op in distractors]
            rng.shuffle(candidates)

            for arm in ARMS:
                if arm == "dump_all":
                    rules = candidates
                elif arm == "selected":
                    rules = select_topk(target["name"], candidates, k)
                elif arm == "random_k":
                    rules = rng.sample(candidates, min(k, len(candidates)))  # control: keep k RANDOM
                else:  # oracle
                    rules = [target["rule_text"]]

                prompt = build_prompt(target["name"], a, b, rules)
                if render_only:
                    response = ""
                else:
                    response = client.generate_with_metrics(prompt, model=model).text
                answer = parse_int(response)
                correct = answer == expected

                rows.append(
                    {
                        "problem_i": problem_i,
                        "op": target["name"],
                        "N": n,
                        "arm": arm,
                        "expected": expected,
                        "answer": answer,
                        "correct": correct,
                    }
                )

    summary = build_summary(rows, context_sizes)
    return {
        "benchmark": "brain-runtime-context-selection",
        "model": model,
        "n_problems": n_problems,
        "context_sizes": list(context_sizes),
        "k": k,
        "seed": seed,
        "render_only": render_only,
        "rows": rows,
        "summary": summary,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def build_summary(rows: list[dict[str, Any]], context_sizes: tuple[int, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for arm in ARMS:
        for n in context_sizes:
            selected = [row for row in rows if row["arm"] == arm and row["N"] == n]
            accuracy = (sum(row["correct"] for row in selected) / len(selected)) if selected else 0.0
            summary[f"{arm}|{n}"] = {
                "arm": arm,
                "N": n,
                "accuracy": accuracy,
                "n": len(selected),
            }
    return summary


def print_matrix(summary: dict[str, Any], context_sizes: tuple[int, ...]) -> None:
    header = "arm".ljust(12) + "".join(f"N={n}".rjust(10) for n in context_sizes)
    print(header)
    for arm in ARMS:
        cells = []
        for n in context_sizes:
            entry = summary[f"{arm}|{n}"]
            cells.append(f"{entry['accuracy']:.2f}".rjust(10))
        print(arm.ljust(12) + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-sizes", default="5,15,30")
    parser.add_argument("--n-problems", type=int, default=20)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-operators", type=int, default=30)
    parser.add_argument("--out", default="results/brain_runtime/context_selection.json")
    parser.add_argument("--render-only", action="store_true", help="Build prompts without calling Ollama.")
    args = parser.parse_args()

    context_sizes = tuple(int(value) for value in args.context_sizes.split(",") if value)
    operators = make_operators(args.n_operators, seed=args.seed)
    client = None if args.render_only else OllamaClient(timeout_seconds=180.0)

    payload = run(
        client,
        model=args.model,
        operators=operators,
        n_problems=args.n_problems,
        context_sizes=context_sizes,
        k=args.k,
        seed=args.seed,
        render_only=args.render_only,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_matrix(payload["summary"], context_sizes)


if __name__ == "__main__":
    main()
