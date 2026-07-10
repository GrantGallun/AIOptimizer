#!/usr/bin/env python3
"""Context selection vs dumping, on REAL CODE (functions = concepts).

HYPOTHESIS (Fable-owned): as a codebase context grows, dumping all functions
degrades (lost-in-the-middle); encoder-selecting the relevant functions holds
accuracy; abstracting non-relevant functions to signatures (compaction) is a
middle ground. The question is buried in one function of a big haystack of
real functions pulled from this repo (not synthetic operators).

Five arms share the same problem:

- dump_all: every candidate function's full source is placed in the prompt.
- selected: only the top-k functions by embedding similarity to the query are
  kept in full (reuses `select_topk` from context_selection.py).
- abstracted: the top-k selected functions keep their full source; every
  other function is compacted to its signature only (`abstract`).
- random_k: k functions chosen uniformly at random are kept in full
  (control: same context size as `selected`, but not query-relevant).
- oracle: only the target function's own source is given (upper bound).
"""

from __future__ import annotations

import ast
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

from experiments.brain_runtime.context_selection import select_topk
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient


ARMS: tuple[str, ...] = ("dump_all", "selected", "abstracted", "random_k", "oracle")


def _constant_value(node: ast.AST | None) -> Any:
    """The literal value of `node` if it's an int/str/float constant, else None."""
    if node is None:
        return None
    if isinstance(node, ast.Constant) and type(node.value) in (int, str, float):
        return node.value
    return None


def _collect_defaults(node: ast.FunctionDef) -> dict[str, Any]:
    """{arg_name: literal_value} for params with an int/str/float default.

    Covers both plain positional-or-keyword defaults (`node.args.args` /
    `node.args.defaults`) and keyword-only defaults (`node.args.kwonlyargs` /
    `node.args.kw_defaults`) since most functions in this repo use the
    `def f(self, *, kw=default)` keyword-only style.
    """
    defaults: dict[str, Any] = {}
    args = node.args

    if args.defaults:
        paired = zip(args.args[-len(args.defaults):], args.defaults)
        for arg, default in paired:
            value = _constant_value(default)
            if value is not None:
                defaults[arg.arg] = value

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        value = _constant_value(default)
        if value is not None:
            defaults[arg.arg] = value

    return defaults


def extract_functions(paths: list[str]) -> list[dict[str, Any]]:
    """Walk `*.py` files under each path (a dir) and collect every function def.

    Returns one dict per function (top-level or nested), deduped by
    (file, name): {file, name, source, signature, defaults}.
    """
    seen: set[tuple[str, str]] = set()
    functions: list[dict[str, Any]] = []

    for base in paths:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for file_path in sorted(base_path.rglob("*.py")):
            try:
                text = file_path.read_text(encoding="utf-8")
                tree = ast.parse(text)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

            file_str = str(file_path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                key = (file_str, node.name)
                if key in seen:
                    continue
                source = ast.get_source_segment(text, node)
                if source is None:
                    continue
                seen.add(key)
                signature = f"{node.name}(" + ", ".join(a.arg for a in node.args.args) + ")"
                functions.append(
                    {
                        "file": file_str,
                        "name": node.name,
                        "source": source,
                        "signature": signature,
                        "defaults": _collect_defaults(node),
                    }
                )
    return functions


def functions_with_defaults(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Functions with at least one captured (int/str/float) default value."""
    return [f for f in functions if f.get("defaults")]


def build_qa(functions: list[dict[str, Any]], seed: int, n: int) -> list[dict[str, Any]]:
    """Sample `n` (param, default value) questions from functions with defaults."""
    candidates = functions_with_defaults(functions)
    if not candidates:
        return []
    rng = random.Random(seed)
    sample_n = min(n, len(candidates))
    chosen = rng.sample(candidates, sample_n)

    qa: list[dict[str, Any]] = []
    for func in chosen:
        param = rng.choice(sorted(func["defaults"].keys()))
        value = func["defaults"][param]
        qa.append(
            {
                "func": func["name"],
                "file": func["file"],
                "param": param,
                "answer": str(value),
                "query": (
                    f"What is the default value of the parameter '{param}' in the "
                    f"function {func['name']}? Reply with only the value."
                ),
            }
        )
    return qa


def abstract(func: dict[str, Any]) -> str:
    """Compact a function to its signature only (the middle-ground arm)."""
    return f"def {func['signature']}: ...  # body omitted"


def render_context(
    target: dict[str, Any],
    haystack: list[dict[str, Any]],
    arm: str,
    k: int,
    qa: dict[str, Any] | None = None,
    rng: random.Random | None = None,
) -> str:
    """Build the code context string for `arm` from `haystack` (includes target)."""
    if arm == "dump_all":
        return "\n\n".join(f["source"] for f in haystack)

    if arm == "oracle":
        return target["source"]

    if arm in ("selected", "abstracted"):
        if qa is not None:
            query = f"{qa['func']} {qa['param']} default"
        else:
            query = target["name"]
        sources = [f["source"] for f in haystack]
        top_sources = select_topk(query, sources, k)
        if arm == "selected":
            return "\n\n".join(top_sources)
        top_set = set(top_sources)
        parts = [f["source"] if f["source"] in top_set else abstract(f) for f in haystack]
        return "\n\n".join(parts)

    if arm == "random_k":
        chooser = rng if rng is not None else random.Random()
        k_actual = min(k, len(haystack))
        chosen = chooser.sample(haystack, k_actual)
        return "\n\n".join(f["source"] for f in chosen)

    raise ValueError(f"Unknown arm: {arm}")


def build_prompt(context: str, query: str) -> str:
    return (
        "You are answering a question about a codebase. Use only the provided code.\n\n"
        "Code:\n" + context + f"\n\nQuestion: {query}\nAnswer:"
    )


def graded(response: str, answer: str) -> bool:
    """Lenient exact-value containment: does `answer` appear as a token in `response`?"""
    target = answer.strip().lower()
    tokens = re.findall(r"[A-Za-z0-9_.'-]+", response.lower())
    return target in tokens


def _find_target(functions: list[dict[str, Any]], item: dict[str, Any]) -> dict[str, Any] | None:
    file = item.get("file")
    name = item["func"]
    if file is not None:
        for f in functions:
            if f["file"] == file and f["name"] == name:
                return f
    for f in functions:
        if f["name"] == name:
            return f
    return None


def run(
    client: OllamaClient | None,
    *,
    model: str = DEFAULT_MODEL,
    functions: list[dict[str, Any]],
    qa: list[dict[str, Any]],
    context_sizes: tuple[int, ...] = (10, 40),
    k: int = 3,
    seed: int = 0,
    render_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for qa_i, item in enumerate(qa):
        target = _find_target(functions, item)
        if target is None:
            continue

        for n in context_sizes:
            pool = [f for f in functions if f["name"] != target["name"]]
            distractor_count = min(max(n - 1, 0), len(pool))
            distractors = rng.sample(pool, distractor_count)
            haystack = [target] + distractors
            rng.shuffle(haystack)

            for arm in ARMS:
                context = render_context(target, haystack, arm, k, item, rng)
                prompt = build_prompt(context, item["query"])
                if render_only:
                    response = ""
                    correct = False
                else:
                    response = client.generate_with_metrics(prompt, model=model).text
                    correct = graded(response, item["answer"])

                rows.append(
                    {
                        "qa_i": qa_i,
                        "func": item["func"],
                        "param": item["param"],
                        "N": n,
                        "arm": arm,
                        "answer": item["answer"],
                        "correct": correct,
                    }
                )

    summary = build_summary(rows, context_sizes)
    return {
        "benchmark": "brain-runtime-code-context",
        "model": model,
        "n_qa": len(qa),
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
    parser.add_argument("--paths", default="agent_bus,experiments/brain_runtime")
    parser.add_argument("--context-sizes", default="10,40")
    parser.add_argument("--n-qa", type=int, default=15)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/brain_runtime/code_context.json")
    parser.add_argument("--render-only", action="store_true", help="Build prompts without calling Ollama.")
    args = parser.parse_args()

    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    context_sizes = tuple(int(value) for value in args.context_sizes.split(",") if value)

    functions = extract_functions(paths)
    qa = build_qa(functions, args.seed, args.n_qa)
    client = None if args.render_only else OllamaClient(timeout_seconds=180.0)

    payload = run(
        client,
        model=args.model,
        functions=functions,
        qa=qa,
        context_sizes=context_sizes,
        k=args.k,
        seed=args.seed,
        render_only=args.render_only,
    )
    payload["paths"] = paths
    payload["n_functions"] = len(functions)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_matrix(payload["summary"], context_sizes)


if __name__ == "__main__":
    main()
