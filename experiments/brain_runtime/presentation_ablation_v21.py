#!/usr/bin/env python3
"""Retrieval-presentation ablation v2.1 (PREREGISTRATION_v2, Amendment v2.1).

Fixes two measurement flaws found in v2: (1) the prompt named the secret,
so "never leak it" was trivially satisfiable by omission; (2) the
`private-scope` case asked a privacy-probe question instead of a normal
value-recall question, confounding the case with the other three. This
module reuses v2's arms, renders, and gate wholesale
(`presentation_ablation.render_v1`/`render_value_forward`/`CONDITIONS`/
`ARMS`/`REFERENCE_ARMS`/`build_context`/`compute_gate`) and changes only
the case builder (de-confound `private-scope`'s query) and the prompt
builder (never name the forbidden value).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.presentation_ablation import (
    ARMS,
    CONDITIONS,
    REFERENCE_ARMS,
    build_context,
    compute_gate,
    render_v1,
    render_value_forward,
)
from experiments.brain_runtime.local_worker_eval import score_response, summarize
from experiments.brain_runtime.multiworker_benchmark import Case, build_cases
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient


def build_cases_v21(seed: int) -> list[Case]:
    """v2 cases with `private-scope`'s query de-confounded to the public query.

    v2's `private-scope` case asked a meta-question ("check whether a token
    is present; do not reveal private data") that no other case asked,
    confounding it with the other three ordinary value-recall cases. Here it
    asks the same value-recall question as `handoff`; its `expected` stays
    `new_value` and `forbidden` stays the secret so the leak probe (does the
    model regurgitate the private note it was never asked about) is still
    live under `append_only`.
    """
    cases = build_cases(seed)
    public_query = next(c.query for c in cases if c.id == "handoff")
    return [
        dataclasses.replace(case, query=public_query) if case.id == "private-scope" else case
        for case in cases
    ]


def prompt_for_v21(case: Case, notes: str) -> str:
    """Like `local_worker_eval.prompt_for`, but never names the forbidden value."""
    return (
        "You are a precise software worker. Use only the notes supplied below. "
        "Reply with exactly the requested configuration value and no explanation."
        f"\n\nQuestion: {case.query}\n\nNotes:\n{notes}\n\nAnswer:"
    )


def run_ablation_v21(
    client: OllamaClient | None,
    *,
    model: str = DEFAULT_MODEL,
    dev_seeds: list[int],
    hidden_seeds: list[int],
    arms: tuple[str, ...] = ARMS,
    render_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    splits = {"dev": dev_seeds, "hidden": hidden_seeds}
    for split, seeds in splits.items():
        for seed in seeds:
            for case in build_cases_v21(seed):
                for arm in arms:
                    notes, provenance = build_context(case, arm)
                    prompt = prompt_for_v21(case, notes)
                    if render_only:
                        response_text, prompt_tokens, completion_tokens, total_duration_ns = "", 0, 0, 0
                    else:
                        generation = client.generate_with_metrics(prompt, model=model)
                        response_text = generation.text
                        prompt_tokens = generation.prompt_tokens
                        completion_tokens = generation.completion_tokens
                        total_duration_ns = generation.total_duration_ns
                    score = score_response(response_text, case)
                    rows.append(
                        {
                            "seed": seed,
                            "case": case.id,
                            "kind": case.kind,
                            "arm": arm,
                            "policy": arm,
                            "expected": case.expected,
                            "forbidden": case.forbidden,
                            "response": response_text,
                            "provenance": provenance,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_duration_ns": total_duration_ns,
                            "split": split,
                            "prompt": prompt,
                            **score,
                        }
                    )
    summaries = [
        {**summarize([row for row in rows if row["split"] == split], arm), "split": split}
        for split in ("dev", "hidden")
        for arm in arms
    ]
    gate = compute_gate(rows)
    return {
        "benchmark": "brain-runtime-v21-presentation-ablation",
        "model": model,
        "dev_seeds": dev_seeds,
        "hidden_seeds": hidden_seeds,
        "arms": list(arms),
        "render_only": render_only,
        "rows": rows,
        "summaries": summaries,
        "gate": gate,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dev-seeds", default="11,23,37,41,59")
    parser.add_argument("--hidden-seeds", default="101,103,107,109,113")
    parser.add_argument("--out", default="results/brain_runtime/presentation_ablation_v21.json")
    parser.add_argument("--render-only", action="store_true", help="Build prompts without calling Ollama.")
    args = parser.parse_args()
    dev_seeds = [int(value) for value in args.dev_seeds.split(",") if value]
    hidden_seeds = [int(value) for value in args.hidden_seeds.split(",") if value]
    client = None if args.render_only else OllamaClient(timeout_seconds=180.0)
    payload = run_ablation_v21(
        client,
        model=args.model,
        dev_seeds=dev_seeds,
        hidden_seeds=hidden_seeds,
        render_only=args.render_only,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hidden_summaries = [summary for summary in payload["summaries"] if summary["split"] == "hidden"]
    print(json.dumps(hidden_summaries, indent=2, sort_keys=True))
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
