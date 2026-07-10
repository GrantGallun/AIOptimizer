#!/usr/bin/env python3
"""Presentation ablation v2.2 — the privacy/value split (design credit: Codex).

v2.1 de-confounded `private-scope` *in place* by turning it into a value-recall question,
which merged two signals into one case: "did the model recall the value" and "did it leak
the private note." Codex's independent v2.1 attempt separated them into two cases; that split
is the more rigorous design. Codex's literal file was lost to a coordination overwrite (a
Sonnet worker rewrote it; the repo had no commits to recover from) — this module reconstructs
that design direction and credits it. See PREREGISTRATION_v2.md and HYP-20260709-12/13.

Five cases per seed (25 hidden tasks): stale-fact, handoff, **value-recall** (pure recall, no
secret present), **privacy-probe** (value recall with an out-of-scope secret distractor; success
requires the value AND no leak), contradiction. Only rendering + this case split change; the
governed retrieval, scorer, and gate are reused unchanged.
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

from experiments.brain_runtime.local_worker_eval import score_response, summarize
from experiments.brain_runtime.multiworker_benchmark import Case
from experiments.brain_runtime.presentation_ablation import ARMS, build_context, compute_gate
from experiments.brain_runtime.presentation_ablation_v21 import build_cases_v21, prompt_for_v21
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient


def build_cases_v22(seed: int) -> list[Case]:
    """v2.1 cases, but split the de-confounded private-scope case into two clean signals."""
    cases: list[Case] = []
    for case in build_cases_v21(seed):
        if case.id != "private-scope":
            cases.append(case)
            continue
        # value-recall: drop the out-of-scope secret note entirely -> pure recall, no leak signal.
        public_claims = tuple(claim for claim in case.claims if claim.scope != "worker-b")
        cases.append(dataclasses.replace(case, id="value-recall", kind="value", forbidden=None, claims=public_claims))
        # privacy-probe: keep the secret note as an out-of-scope distractor; success = value AND no leak.
        cases.append(dataclasses.replace(case, id="privacy-probe", kind="privacy"))
    return cases


def run_ablation_v22(
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
    for split, seeds in {"dev": dev_seeds, "hidden": hidden_seeds}.items():
        for seed in seeds:
            for case in build_cases_v22(seed):
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
    return {
        "benchmark": "brain-runtime-v22-presentation-ablation-privacy-value-split",
        "design_credit": "Codex (privacy/value split); reconstructed by Fable after an overwrite",
        "model": model,
        "dev_seeds": dev_seeds,
        "hidden_seeds": hidden_seeds,
        "arms": list(arms),
        "render_only": render_only,
        "rows": rows,
        "summaries": summaries,
        "gate": compute_gate(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dev-seeds", default="11,23,37,41,59")
    parser.add_argument("--hidden-seeds", default="101,103,107,109,113")
    parser.add_argument("--out", default="results/brain_runtime/presentation_ablation_v22.json")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    dev_seeds = [int(v) for v in args.dev_seeds.split(",") if v]
    hidden_seeds = [int(v) for v in args.hidden_seeds.split(",") if v]
    client = None if args.render_only else OllamaClient(timeout_seconds=180.0)
    payload = run_ablation_v22(client, model=args.model, dev_seeds=dev_seeds, hidden_seeds=hidden_seeds, render_only=args.render_only)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps([s for s in payload["summaries"] if s["split"] == "hidden"], indent=2, sort_keys=True))
    print(json.dumps(payload["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
