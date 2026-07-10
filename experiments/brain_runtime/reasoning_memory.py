#!/usr/bin/env python3
"""Does an external memory loop let a model learn across a session?

A sequence of arithmetic problems uses made-up operators (novel names, so the
model cannot already know the rule). Each operator recurs several times. On an
operator's FIRST appearance the model has to guess (no memory can help); on
RECURRENCES, a model that remembers the rule it inferred (or was told) earlier
in the session should compute correctly. Three arms are compared over the same
problem sequence:

- raw: no memory at all.
- append_only: every rule learned so far is dumped into the prompt.
- brain: BrainRuntime retrieves only the rule relevant to the current operator.

The headline metric is accuracy on RECURRENCES (first_appearance == False):
memory should help brain and append_only there, but cannot help raw, since raw
never receives any rule regardless of repetition.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.runtime import BrainRuntime
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient


OPERATORS = {
    "glorp": (lambda a, b: a + b + 3, "glorp(a,b) = a + b + 3"),
    "zib": (lambda a, b: a * 2 + b, "zib(a,b) = (a times 2) + b"),
    "frotz": (lambda a, b: a + b * 2, "frotz(a,b) = a + (b times 2)"),
    "quan": (lambda a, b: abs(a - b) + 5, "quan(a,b) = |a - b| + 5"),
    "vlim": (lambda a, b: max(a, b) * 2, "vlim(a,b) = 2 times max(a, b)"),
}

ARMS = ("raw", "append_only", "brain")

INT_RE = re.compile(r"-?\d+")


def build_sequence(seed: int, reps: int = 5) -> list[dict]:
    rng = random.Random(seed)
    occurrences: list[str] = []
    for name in OPERATORS:
        occurrences.extend([name] * reps)
    rng.shuffle(occurrences)

    seen: set[str] = set()
    sequence: list[dict] = []
    for step, op in enumerate(occurrences):
        fn, _rule_text = OPERATORS[op]
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        first_appearance = op not in seen
        seen.add(op)
        sequence.append(
            {
                "step": step,
                "op": op,
                "a": a,
                "b": b,
                "expected": fn(a, b),
                "first_appearance": first_appearance,
            }
        )
    return sequence


def parse_int(text: str) -> int | None:
    match = INT_RE.search(text)
    if match is None:
        return None
    return int(match.group())


def build_prompt(problem: dict, lessons: list[str]) -> str:
    prefix = "Compute the result. Reply with ONLY an integer, nothing else.\n\n"
    if lessons:
        prefix += "Rules you must apply exactly:\n" + "\n".join(f"- {lesson}" for lesson in lessons) + "\n\n"
    else:
        prefix += "No rule is given; make your best guess.\n\n"
    return prefix + f"{problem['op']}({problem['a']}, {problem['b']}) = ?"


def run(
    client: OllamaClient,
    *,
    model: str = DEFAULT_MODEL,
    seeds: list[int],
    reps: int = 5,
    render_only: bool = False,
) -> dict:
    started = time.perf_counter()
    rows: list[dict] = []

    for seed in seeds:
        sequence = build_sequence(seed, reps=reps)
        for arm in ARMS:
            learned: list[str] = []
            runtime = BrainRuntime() if arm == "brain" else None

            for problem in sequence:
                op = problem["op"]
                _fn, rule_text = OPERATORS[op]

                if arm == "raw":
                    lessons: list[str] = []
                elif arm == "append_only":
                    lessons = list(learned)
                else:
                    lessons = [item.content for item in runtime.retrieve(op, limit=2)]

                prompt = build_prompt(problem, lessons)
                response = "" if render_only else client.generate(prompt, model=model)
                parsed = parse_int(response)
                correct = parsed == problem["expected"]

                if arm == "append_only":
                    if rule_text not in learned:
                        learned.append(rule_text)
                elif arm == "brain":
                    runtime.remember(topic=op, content=rule_text, key=op, value=rule_text)

                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "step": problem["step"],
                        "op": op,
                        "first_appearance": problem["first_appearance"],
                        "expected": problem["expected"],
                        "answer": parsed,
                        "correct": correct,
                        "n_lessons": len(lessons),
                    }
                )

    summaries = [summarize(rows, arm) for arm in ARMS]
    return {
        "benchmark": "brain-runtime-reasoning-memory-v1",
        "model": model,
        "seeds": seeds,
        "reps": reps,
        "arms": list(ARMS),
        "rows": rows,
        "summaries": summaries,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def summarize(rows: list[dict], arm: str) -> dict:
    selected = [row for row in rows if row["arm"] == arm]
    first = [row for row in selected if row["first_appearance"]]
    recurrence = [row for row in selected if not row["first_appearance"]]
    return {
        "arm": arm,
        "n": len(selected),
        "overall_acc": _accuracy(selected),
        "first_appearance_acc": _accuracy(first),
        "recurrence_acc": _accuracy(recurrence),
        "mean_lessons": (sum(row["n_lessons"] for row in selected) / len(selected)) if selected else 0.0,
    }


def _accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(row["correct"] for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seeds", default="101,103,107")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", default="results/brain_runtime/reasoning_memory.json")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value]
    client = OllamaClient(timeout_seconds=180.0)
    payload = run(client, model=args.model, seeds=seeds, reps=args.reps, render_only=args.render_only)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'arm':<12} {'overall':>8} {'first-appear':>13} {'RECURRENCE':>11} {'mean_lessons':>13}")
    for summary in payload["summaries"]:
        print(
            f"{summary['arm']:<12} {summary['overall_acc']:>8.3f} {summary['first_appearance_acc']:>13.3f} "
            f"{summary['recurrence_acc']:>11.3f} {summary['mean_lessons']:>13.3f}"
        )


if __name__ == "__main__":
    main()
