"""v17 runner: budget sweep at N=320, per PREREGISTRATION_v17.md.

Reuses the frozen v16 generator and the frozen context_organization_eval harness. ONLY
budget_chars varies. Dev sanity must reproduce v16's shape before any hidden read.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"c:\Code\AIOptimizer")

from experiments.brain_runtime.make_context_cases import generate_volume_cases
from experiments.brain_runtime.context_organization_eval import run_evaluation
from experiments.local_worker.ollama_client import OllamaClient
from experiments.brain_runtime.stats import wilson_interval

N_MESSAGES = 320
BUDGETS = (2600, 1300, 650)
OUT = Path(r"c:\Code\AIOptimizer\results\brain_runtime")


def run(seed_label: str, seed: int, budgets=BUDGETS):
    table = {}
    for budget in budgets:
        cases = generate_volume_cases(seed, n_cases=40, n_messages=N_MESSAGES, budget_chars=budget)
        result = run_evaluation(cases, OllamaClient('http://127.0.0.1:11434', timeout_seconds=180.0))
        rows = result["rows"]
        path = OUT / f"v17_org_N320_b{budget}_{seed_label}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        by_arm = defaultdict(list)
        for r in rows:
            by_arm[r["arm"]].append(r)
        for arm, rs in by_arm.items():
            succ = sum(1 for r in rs if r.get("success"))
            table[(budget, arm)] = {
                "n": len(rs), "successes": succ, "rate": succ / len(rs),
                "prompt_tokens": statistics.fmean(r.get("prompt_tokens", 0) for r in rs),
            }
    return table


def show(label, table):
    print(f"\n{label}")
    print(f"  {'budget':>7} {'arm':<11} {'n':>3} {'success':>8} {'wilson95':>16} {'prompt_tok':>11}")
    for (budget, arm) in sorted(table):
        e = table[(budget, arm)]
        lo, hi = wilson_interval(e["successes"], e["n"])
        print(f"  {budget:>7} {arm:<11} {e['n']:>3} {e['rate']:>8.3f} "
              f"  [{lo:.3f}, {hi:.3f}] {e['prompt_tokens']:>11.0f}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dev", "hidden"], required=True)
    args = ap.parse_args()

    if args.stage == "dev":
        # Gate: must reproduce v16's shape at budget 2600 or STOP.
        t = show("DEV SANITY (seed 20260711, N=320)", run("dev", 20260711, budgets=(2600,)))
        att = t[(2600, "attention")]["rate"]
        raw = t[(2600, "raw")]["rate"]
        print(f"\n  v16 hidden reference @2600/N320: raw 0.567, attention 1.000")
        print(f"  dev reproduction:                 raw {raw:.3f}, attention {att:.3f}")
        ok = att >= 0.9 and raw < att - 0.15
        print(f"\n  [{'PASS' if ok else 'STOP'}] harness reproduces v16's shape"
              f"{'' if ok else ' — NO hidden read permitted; explain the drift first'}")
    else:
        merged = defaultdict(lambda: {"n": 0, "successes": 0, "pt": []})
        for seed in (1301, 1303, 1307):
            t = run(f"h{seed}", seed)
            for key, e in t.items():
                merged[key]["n"] += e["n"]
                merged[key]["successes"] += e["successes"]
                merged[key]["pt"].append(e["prompt_tokens"])
        table = {k: {"n": v["n"], "successes": v["successes"], "rate": v["successes"] / v["n"],
                     "prompt_tokens": statistics.fmean(v["pt"])} for k, v in merged.items()}
        show("HIDDEN (v17 seeds 1301/1303/1307 pooled) — READ ONCE", table)
        a = table[(1300, "attention")]
        r = table[(2600, "raw")]
        a_lo, _ = wilson_interval(a["successes"], a["n"])
        _, r_hi = wilson_interval(r["successes"], r["n"])
        cheaper = a["prompt_tokens"] < r["prompt_tokens"]
        print("\n  GATE H-v17a: wilson_lower(attention@1300) > wilson_upper(raw@2600) AND cheaper")
        print(f"    wilson_lower(attention@1300) = {a_lo:.3f}")
        print(f"    wilson_upper(raw@2600)       = {r_hi:.3f}")
        print(f"    prompt_tokens {a['prompt_tokens']:.0f} vs {r['prompt_tokens']:.0f} -> cheaper={cheaper}")
        print(f"\n  => H-v17a {'CONFIRMED' if (a_lo > r_hi and cheaper) else 'NOT CONFIRMED'}")
