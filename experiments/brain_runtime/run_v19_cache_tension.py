#!/usr/bin/env python3
"""v19: does per-turn reorganization erase the provider's prefix-cache benefit?

    python experiments/brain_runtime/run_v19_cache_tension.py --stage dev
    python experiments/brain_runtime/run_v19_cache_tension.py --stage hidden   # read ONCE

Reuses frozen v16-v18 pieces (generate_volume_cases, ConversationCompiler) without mutating
them. Measures Ollama's own prompt_eval_duration per checkpoint for two arms built from a
growing conversation: raw (strict textual append, cache-friendly by construction) vs attention
(recompiled/reorganized every checkpoint). See PREREGISTRATION_v19.md for the paired sign-test
gate and the mechanism probe that motivated this design.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aioptimizer.context_compiler import ConversationCompiler
from experiments.brain_runtime.make_context_cases import generate_volume_cases
from experiments.brain_runtime.stats import FRESH_HIDDEN_SEEDS
from experiments.local_worker.ollama_client import OllamaClient

N_MESSAGES = 320
N_CASES = 5
CHECKPOINT_STEP = 40
BUDGET_CHARS = 2600
NUM_CTX = 16384
MODEL = "qwen3:8b"
DEV_SEED = 20260711
HIDDEN_SEEDS = FRESH_HIDDEN_SEEDS["v19"]  # (1429, 1433, 1439) — read once.
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "brain_runtime"


def render_plain(messages: list[dict[str, str]]) -> str:
    """The transcript as an agent would send it: append-only, chronological."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def checkpoints(n_messages: int, step: int) -> list[int]:
    return list(range(step, n_messages + 1, step))


def build_case_prompts(case: dict[str, Any], compiler: ConversationCompiler) -> dict[str, list[str]]:
    """Per checkpoint k, the raw and attention prompts built from messages[:k]."""
    raw_prompts, attention_prompts = [], []
    for k in checkpoints(len(case["messages"]) - 1, CHECKPOINT_STEP):  # -1: exclude the final query line
        prefix_messages = case["messages"][:k]
        raw_prompts.append(render_plain(prefix_messages))
        compiled = compiler.compile(prefix_messages)
        organized = compiler.organize(compiled, query=str(case["query"]), max_records=len(compiled.records))
        attention_prompts.append(compiler.render_organized(organized, budget_chars=BUDGET_CHARS))
    return {"raw": raw_prompts, "attention": attention_prompts}


def mechanical_sanity(seed: int) -> dict[str, Any]:
    """Prereg dev-sanity step 1: raw checkpoints must be strict prefix extensions. No model calls."""
    compiler = ConversationCompiler()
    cases = generate_volume_cases(seed, n_cases=N_CASES, n_messages=N_MESSAGES)
    violations = []
    for case in cases:
        prompts = build_case_prompts(case, compiler)["raw"]
        for prev, cur in zip(prompts, prompts[1:]):
            if not cur.startswith(prev):
                violations.append(case["id"])
                break
    return {"n": len(cases), "violations": violations, "passed": not violations}


def run_seed(seed: int, seed_label: str, client: OllamaClient) -> list[dict[str, Any]]:
    compiler = ConversationCompiler()
    cases = generate_volume_cases(seed, n_cases=N_CASES, n_messages=N_MESSAGES)
    rows = []
    for case in cases:
        prompts = build_case_prompts(case, compiler)
        for arm in ("raw", "attention"):  # sequential per arm, never interleaved: protects the KV-cache slot
            for idx, prompt in enumerate(prompts[arm]):
                started = time.perf_counter()
                gen = client.generate_with_metrics(
                    prompt, model=MODEL, temperature=0.0, max_tokens=1, num_ctx=NUM_CTX,
                )
                wall = time.perf_counter() - started
                rows.append({
                    "seed": seed, "case": case["id"], "arm": arm, "checkpoint_index": idx,
                    "prompt_chars": len(prompt),
                    "prompt_tokens": gen.prompt_tokens,
                    "prompt_eval_ms": round(gen.prompt_eval_duration_ns / 1e6, 3),
                    "wall_seconds": round(wall, 6),
                })
    out = OUT_DIR / f"v19_cache_tension_{seed_label}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing result: {out}")
    out.write_text(json.dumps({"benchmark": "v19-cache-tension", "model": MODEL,
                               "num_ctx": NUM_CTX, "rows": rows}, indent=2) + "\n",
                   encoding="utf-8")
    return rows


def per_case_totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        key = f"{row['seed']}:{row['case']}"
        entry = totals.setdefault(key, {"raw_ms": 0.0, "attention_ms": 0.0,
                                        "raw_tokens": 0, "attention_tokens": 0})
        entry[f"{row['arm']}_ms"] += row["prompt_eval_ms"]
        entry[f"{row['arm']}_tokens"] += row["prompt_tokens"]
    return totals


def show_dev(rows: list[dict[str, Any]]) -> bool:
    for arm in ("raw", "attention"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        by_case: dict[str, list[float]] = {}
        for r in arm_rows:
            by_case.setdefault(r["case"], []).append(r["prompt_eval_ms"])
        cold = [v[0] for v in by_case.values()]
        warm = [ms for v in by_case.values() for ms in v[1:]]
        print(f"  {arm:<10} checkpoint-1 (cold) median={statistics.median(cold):.1f}ms  "
              f"checkpoint-2..N median={statistics.median(warm):.1f}ms  "
              f"ratio={statistics.median(warm) / statistics.median(cold):.2f}")
    p1_p2 = {}
    for arm, predict_low_ratio in (("raw", True), ("attention", False)):
        arm_rows = [r for r in rows if r["arm"] == arm]
        by_case: dict[str, list[float]] = {}
        for r in arm_rows:
            by_case.setdefault(r["case"], []).append(r["prompt_eval_ms"])
        ratios = [statistics.median(v[1:]) / v[0] for v in by_case.values() if v[0] > 0]
        median_ratio = statistics.median(ratios)
        ok = (median_ratio < 0.30) if predict_low_ratio else (median_ratio > 0.70)
        p1_p2[arm] = ok
        label = "P1 raw drops sharply" if arm == "raw" else "P2 attention stays high"
        print(f"  [{'PASS' if ok else 'STOP'}] {label} (median warm/cold ratio={median_ratio:.2f})")
    return all(p1_p2.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dev", "hidden"], required=True)
    args = parser.parse_args()
    client = OllamaClient("http://127.0.0.1:11434", timeout_seconds=300.0)

    if args.stage == "dev":
        sanity = mechanical_sanity(DEV_SEED)
        print(f"MECHANICAL SANITY (no model): {sanity['n']} cases, "
              f"violations={sanity['violations']}")
        print(f"  [{'PASS' if sanity['passed'] else 'STOP'}] raw checkpoints are strict prefix extensions")
        if not sanity["passed"]:
            raise SystemExit("STOP: geometry check failed; no model run, no hidden read.")
        rows = run_seed(DEV_SEED, "dev", client)
        print("\nDEV (seed 20260711):")
        passed = show_dev(rows)
        print(f"\n  [{'PASS' if passed else 'STOP'}] P1+P2 mechanism reproduces on harness prompt shapes")
        return

    print("HIDDEN stage: seeds", HIDDEN_SEEDS, "— read-once event; results are binding.")
    rows = []
    for seed in HIDDEN_SEEDS:
        rows.extend(run_seed(seed, f"h{seed}", client))
    totals = per_case_totals(rows)
    n = len(totals)
    attention_slower = sum(1 for e in totals.values() if e["attention_ms"] > e["raw_ms"])
    attention_faster = sum(1 for e in totals.values() if e["attention_ms"] < e["raw_ms"])

    print(f"\nHIDDEN (pooled, n={n} cases):")
    for key, e in totals.items():
        print(f"  {key:<30} raw={e['raw_ms']:>8.1f}ms ({int(e['raw_tokens'])} tok)   "
              f"attention={e['attention_ms']:>8.1f}ms ({int(e['attention_tokens'])} tok)")
    raw_ms_med = statistics.median(e["raw_ms"] for e in totals.values())
    att_ms_med = statistics.median(e["attention_ms"] for e in totals.values())
    raw_tok_med = statistics.median(e["raw_tokens"] for e in totals.values())
    att_tok_med = statistics.median(e["attention_tokens"] for e in totals.values())
    print(f"\n  median totals: raw {raw_ms_med:.1f}ms / {raw_tok_med:.0f} tok   "
          f"attention {att_ms_med:.1f}ms / {att_tok_med:.0f} tok")
    print(f"  attention_total_ms > raw_total_ms in {attention_slower}/{n} cases "
          f"({attention_slower / n:.0%})")
    print(f"  attention_total_ms < raw_total_ms in {attention_faster}/{n} cases "
          f"({attention_faster / n:.0%})")

    # Amendment v19.1: secondary, non-gating metric — how much of the token-count-implied
    # time savings the cache actually erases (1 = no erosion, 0 = cache erases it entirely).
    compressions = []
    for e in totals.values():
        time_savings = 1 - e["attention_ms"] / e["raw_ms"]
        token_savings = 1 - e["attention_tokens"] / e["raw_tokens"]
        if token_savings > 0:
            compressions.append(time_savings / token_savings)
    print(f"\n  dividend-compression ratio (time-savings / token-savings), median="
          f"{statistics.median(compressions):.2f} range=[{min(compressions):.2f}, "
          f"{max(compressions):.2f}]  (1.0=no erosion, 0.0=cache erases the win entirely)")

    if attention_slower / n >= 0.80:
        verdict = "CONFIRMED: attention is the compute-time loser despite fewer tokens"
    elif attention_faster / n >= 0.80:
        verdict = "REFUTED: reorg cost does not materialize on this backend/model"
    else:
        verdict = "INCONCLUSIVE: no clear majority either way"
    print(f"\n  H-v19 -> {verdict}")


if __name__ == "__main__":
    main()
