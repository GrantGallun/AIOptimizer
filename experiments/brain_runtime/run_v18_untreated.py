#!/usr/bin/env python3
"""v18: attention vs genuinely UNTREATED transcripts, per PREREGISTRATION_v18.md.

    python experiments/brain_runtime/run_v18_untreated.py --stage dev
    python experiments/brain_runtime/run_v18_untreated.py --stage hidden   # read ONCE

Composes frozen pieces (generate_volume_cases, prompt_for, score_response, SYSTEM)
without mutating them. Four arms; num_ctx=16384 uniformly; only context construction
varies. See the prereg for arms, predictions, and the committed three-way H-v18b gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aioptimizer.context_compiler import ConversationCompiler
from experiments.brain_runtime.context_organization_eval import (
    SYSTEM,
    prompt_for,
    score_response,
)
from experiments.brain_runtime.make_context_cases import generate_volume_cases
from experiments.brain_runtime.stats import FRESH_HIDDEN_SEEDS, wilson_interval
from experiments.local_worker.ollama_client import OllamaClient

N_MESSAGES = 320
N_CASES = 40
BUDGET_CHARS = 2600
NUM_CTX = 16384
DEV_SEED = 20260711
HIDDEN_SEEDS = FRESH_HIDDEN_SEEDS["v18"]  # (1409, 1423, 1427) — read once.
ARMS = ("untreated_full", "untreated_tail", "attention", "raw_compiler")
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "brain_runtime"


def render_plain(messages: list[dict[str, str]]) -> str:
    """The transcript as an agent would send it: chronological role-prefixed lines."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def render_plain_tail(messages: list[dict[str, str]], budget_chars: int) -> str:
    """Drop oldest messages at message boundaries until the rendering fits the budget."""
    kept = list(messages)
    while kept and len(render_plain(kept)) > budget_chars:
        kept.pop(0)
    return render_plain(kept)


def render_case_arms(case: dict[str, Any], compiler: ConversationCompiler) -> dict[str, str]:
    compiled = compiler.compile(case["messages"])
    organized = compiler.organize(
        compiled, query=str(case["query"]), max_records=len(compiled.records)
    )
    return {
        "untreated_full": render_plain(case["messages"]),
        "untreated_tail": render_plain_tail(case["messages"], BUDGET_CHARS),
        "attention": compiler.render_organized(organized, budget_chars=BUDGET_CHARS),
        "raw_compiler": compiler.render_raw(compiled, budget_chars=BUDGET_CHARS),
    }


def mechanical_sanity(seed: int) -> dict[str, Any]:
    """Prereg dev-sanity step 1: fact presence per arm, no model calls."""
    compiler = ConversationCompiler()
    cases = generate_volume_cases(seed, n_cases=N_CASES, n_messages=N_MESSAGES,
                                  budget_chars=BUDGET_CHARS)
    presence = {arm: 0 for arm in ARMS}
    for case in cases:
        contexts = render_case_arms(case, compiler)
        for arm, context in contexts.items():
            if str(case["expected"]) in context:
                presence[arm] += 1
    n = len(cases)
    report = {arm: count / n for arm, count in presence.items()}
    checks = {
        "untreated_full_contains_fact_always": report["untreated_full"] == 1.0,
        "untreated_tail_lacks_fact_90pct": (1.0 - report["untreated_tail"]) >= 0.9,
    }
    return {"n": n, "presence": report, "checks": checks, "passed": all(checks.values())}


def run_seed(seed: int, seed_label: str, client: OllamaClient) -> list[dict[str, Any]]:
    compiler = ConversationCompiler()
    cases = generate_volume_cases(seed, n_cases=N_CASES, n_messages=N_MESSAGES,
                                  budget_chars=BUDGET_CHARS)
    rows = []
    for case in cases:
        contexts = render_case_arms(case, compiler)
        for arm in ARMS:
            started = time.perf_counter()
            generation = client.generate_with_metrics(
                prompt_for(case, contexts[arm]),
                system=SYSTEM,
                temperature=0.0,
                max_tokens=32,
                num_ctx=NUM_CTX,
            )
            wall = time.perf_counter() - started
            score = score_response(generation.text, case)
            rows.append({
                "seed": seed, "case": case["id"], "arm": arm,
                "context_chars": len(contexts[arm]),
                "context_expected_present": str(case["expected"]) in contexts[arm],
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "wall_seconds": round(wall, 6),
                "response": generation.text,
                **score,
            })
    out = OUT_DIR / f"v18_untreated_N320_{seed_label}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing result: {out}")
    out.write_text(json.dumps({"benchmark": "v18-untreated-baselines", "model": "qwen3:8b",
                               "num_ctx": NUM_CTX, "rows": rows}, indent=2) + "\n",
                   encoding="utf-8")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    table = {}
    for arm in ARMS:
        arm_rows = by_arm[arm]
        # Amendment v18.1: primary endpoint is answer_correct (fair across arms);
        # success (value+attribution) is a secondary product-capability metric.
        successes = sum(1 for r in arm_rows if r.get("answer_correct"))
        lo, hi = wilson_interval(successes, len(arm_rows))
        attributed = sum(1 for r in arm_rows if r.get("success"))
        table[arm] = {
            "n": len(arm_rows), "successes": successes,
            "attributed": attributed,
            "rate": successes / len(arm_rows) if arm_rows else 0.0,
            "wilson": (round(lo, 3), round(hi, 3)),
            "prompt_tokens": round(statistics.fmean(r["prompt_tokens"] for r in arm_rows)),
            "wall_seconds": round(statistics.fmean(r["wall_seconds"] for r in arm_rows), 2),
        }
    return table


def show(table: dict[str, dict[str, Any]]) -> None:
    print(f"  {'arm':<16} {'n':>4} {'answer':>8} {'wilson95':>17} {'prompt_tok':>11} {'sec':>6}")
    for arm, entry in table.items():
        lo, hi = entry["wilson"]
        print(f"  {arm:<16} {entry['n']:>4} {entry['rate']:>8.3f}   [{lo:.3f}, {hi:.3f}] "
              f"{entry['prompt_tokens']:>11} {entry['wall_seconds']:>6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dev", "hidden"], required=True)
    args = parser.parse_args()
    client = OllamaClient("http://127.0.0.1:11434", timeout_seconds=300.0)

    if args.stage == "dev":
        sanity = mechanical_sanity(DEV_SEED)
        print("MECHANICAL SANITY (no model):", json.dumps(sanity["presence"], indent=None))
        for name, ok in sanity["checks"].items():
            print(f"  [{'PASS' if ok else 'STOP'}] {name}")
        if not sanity["passed"]:
            raise SystemExit("STOP: geometry checks failed; no model run, no hidden read.")
        rows = run_seed(DEV_SEED, "dev", client)
        table = summarize(rows)
        print("\nDEV (seed 20260711):")
        show(table)
        att_ok = table["attention"]["rate"] >= 0.9
        print(f"\n  [{'PASS' if att_ok else 'STOP'}] attention >= 0.9 (prereg dev gate)")
        return

    print("HIDDEN stage: seeds", HIDDEN_SEEDS, "— read-once event; results are binding.")
    rows = []
    for seed in HIDDEN_SEEDS:
        rows.extend(run_seed(seed, f"h{seed}", client))
    table = summarize(rows)
    print("\nHIDDEN (pooled, n=120/arm):")
    show(table)

    att, full, tail = table["attention"], table["untreated_full"], table["untreated_tail"]
    a_lo, a_hi = att["wilson"]
    f_lo, f_hi = full["wilson"]
    t_lo, t_hi = tail["wilson"]

    print("\n  H-v18a (vs truncating baseline): CONFIRMED iff "
          f"wilson_lower(attention)={a_lo:.3f} > wilson_upper(untreated_tail)={t_hi:.3f}")
    print(f"    -> {'CONFIRMED' if a_lo > t_hi else 'NOT CONFIRMED'}")

    print("\n  H-v18b (vs full-window no-product), pre-committed three-way:")
    overlap = not (a_lo > f_hi or f_lo > a_hi)
    if overlap and f_lo >= 0.9:
        verdict = "PARITY: same quality at a fraction of the tokens vs doing nothing"
    elif a_lo > f_hi:
        verdict = "QUALITY WIN even when the window fits"
    elif f_lo > a_hi:
        verdict = "PRODUCT HURTS vs doing nothing when the window fits"
    else:
        verdict = "INCONCLUSIVE under the committed rules (CIs overlap, full below 0.9)"
    print(f"    attention [{a_lo:.3f},{a_hi:.3f}] vs untreated_full [{f_lo:.3f},{f_hi:.3f}]"
          f" -> {verdict}")
    print(f"    token note (by construction, not a gate): attention {att['prompt_tokens']} "
          f"vs untreated_full {full['prompt_tokens']} prompt tokens")


if __name__ == "__main__":
    main()
