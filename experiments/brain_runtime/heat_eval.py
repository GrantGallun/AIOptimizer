#!/usr/bin/env python3
"""Prereg v11 runner: usage-heat protection vs query-relevance organization.

Three arms over the frozen codename-mapping cases:
- ``raw``: chronological truncation (reference).
- ``attention``: the gate-passed HYP-33 organization (query-ranked clusters).
- ``attention_heat``: same clusters, re-ranked by a frozen blend
  (0.5 * rank-relevance + 0.5 * normalized cluster heat), where heat is
  ``ConversationCompiler.record_heat`` accumulated over all USER turns except the
  final query (using it would smuggle current-query relevance into the prior).

Rendering, scoring, and the leak/attribution machinery are reused unchanged from
``context_organization_eval``. Cold-mapping retention is reported per arm — the
selectivity control the v11 gate requires.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.context_organization_eval import (
    SYSTEM,
    prompt_for,
    score_response,
    validate_case,
)
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient
from gateway.context_compiler import ConversationCompiler

ARMS = ("raw", "attention", "attention_heat")
HEAT_WEIGHT = 0.5  # frozen in PREREGISTRATION_v11


def _heat_with_self_exclusion(compiler: ConversationCompiler, compiled, case: Mapping[str, Any]) -> dict[str, float]:
    """Heat = accumulated record/query cosine over user turns except the final query,
    EXCLUDING each record's own turn (a turn self-matches at cosine 1.0, which otherwise
    degenerates heat into 'keep the user's own chatter' — caught by the render check)."""
    from gateway.context_compiler import _as_vector, _cosine

    # v11.3 (dated, pre-model): INTRODUCTION-REFERENCE heat. Dense-cosine and flat
    # IDF-overlap heat both failed at render level (see the prereg design log) because
    # they reward similarity, not reference. A foundational record is one that
    # INTRODUCES a rare term which LATER turns go on to use (the antecedent/definition
    # signal): heat = sum over first-introduced rare tokens of idf * later-use count.
    # Cold definitions (introduced, never used again) get zero heat for free.
    import math
    import re as _re

    def _tokens(text: str) -> set[str]:
        return {t for t in _re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}

    turns = [t for t in compiled.turns if t["role"] in ("user", "assistant")]
    final_id = turns[-1]["id"] if turns else None
    turn_tokens = [(t["id"], _tokens(t["content"])) for t in turns]
    doc_freq: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    for tid, toks in turn_tokens:
        for tok in toks:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
            first_seen.setdefault(tok, tid)
    n_turns = max(1, len(turns))

    def idf(tok: str) -> float:
        return math.log(n_turns / doc_freq.get(tok, n_turns))

    later_use: dict[str, int] = {}
    for tid, toks in turn_tokens:
        if tid == final_id:
            continue  # the active query must not feed the prior
        for tok in toks:
            if first_seen.get(tok) != tid:
                later_use[tok] = later_use.get(tok, 0) + 1

    heat: dict[str, float] = {}
    for record in compiled.records:
        sources = set(record.get("source_ids", []))
        rtoks = _tokens(str(record["text"]))
        heat[str(record["id"])] = sum(
            idf(tok) * later_use.get(tok, 0)
            for tok in rtoks
            if first_seen.get(tok) in sources  # this record INTRODUCED the term
        )
    return heat


PROTECT_TOP_K = 2  # v11.3 frozen: protected-set size


def _protect_by_heat(organized: dict[str, Any], heat: Mapping[str, float]) -> dict[str, Any]:
    """v11.3: protected-set semantics — the user's original phrasing ('keep the entire
    thing'). The top-K heat records are PINNED alongside system/query; relevance ranks
    the remainder unchanged. Blending (v11.1/.2) displaced instead of protecting."""
    pinned = list(organized.get("pinned", []))
    pinned_ids = {str(r.get("id")) for r in pinned}
    clusters = [dict(c) for c in organized.get("clusters", [])]
    candidates = [
        r for c in clusters for r in c.get("records", [])
        if float(heat.get(str(r.get("id")), 0.0)) > 0.0 and str(r.get("id")) not in pinned_ids
    ]
    protected = sorted(candidates, key=lambda r: -float(heat.get(str(r.get("id")), 0.0)))[:PROTECT_TOP_K]
    protected_ids = {str(r.get("id")) for r in protected}
    out = dict(organized)
    out["pinned"] = pinned + protected
    out["clusters"] = [
        {**c, "records": [r for r in c.get("records", []) if str(r.get("id")) not in protected_ids]}
        for c in clusters
    ]
    return out


def render_arms_v11(case: Mapping[str, Any], compiler: ConversationCompiler):
    case = validate_case(case)
    compiled = compiler.compile(case["messages"])
    organized = compiler.organize(compiled, query=str(case["query"]), max_records=len(compiled.records))
    heat = _heat_with_self_exclusion(compiler, compiled, case)
    budget = case["budget_chars"]
    contexts = {
        "raw": compiler.render_raw(compiled, budget_chars=budget),
        "attention": compiler.render_organized(organized, budget_chars=budget),
        "attention_heat": compiler.render_organized(_protect_by_heat(organized, heat), budget_chars=budget),
    }
    cold = case.get("cold_mappings", [])
    diagnostics = {
        arm: {
            "context_chars": len(context),
            "expected_present": str(case["expected"]) in context,
            "forbidden_present": any(v in context for v in case["forbidden"]),
            "cold_mappings_present": sum(c in context for c in cold),
        }
        for arm, context in contexts.items()
    }
    return contexts, diagnostics


def run(cases: list[dict[str, Any]], *, model: str, endpoint: str | None, render_only: bool):
    started = time.perf_counter()
    compiler = ConversationCompiler()
    client = None if render_only else OllamaClient(endpoint=endpoint, timeout_seconds=240.0)
    rows = []
    for case in cases:
        contexts, diagnostics = render_arms_v11(case, compiler)
        for arm in ARMS:
            row = {"case": case["id"], "arm": arm, **diagnostics[arm]}
            if not render_only:
                text = client.generate_with_metrics(
                    prompt_for(case, contexts[arm]), model=model, system=SYSTEM,
                    temperature=0.0, max_tokens=64,
                ).text
                row.update(score_response(text, case))
            rows.append(row)
    summaries = {}
    for arm in ARMS:
        sel = [r for r in rows if r["arm"] == arm]
        n = len(sel)
        summaries[arm] = {
            "arm": arm,
            "n": n,
            "expected_present": sum(r["expected_present"] for r in sel) / n,
            "cold_mappings_present_mean": sum(r["cold_mappings_present"] for r in sel) / n,
            "forbidden_present": sum(r["forbidden_present"] for r in sel) / n,
            **({"success": sum(bool(r.get("success")) for r in sel) / n,
                "source_correct": sum(bool(r.get("source_correct")) for r in sel) / n,
                "leaked": sum(bool(r.get("leaked")) for r in sel) / n} if not render_only else {}),
        }
    return {"benchmark": "usage-heat-v11", "model": model, "heat_weight": HEAT_WEIGHT,
            "rows": rows, "summaries": summaries,
            "elapsed_seconds": round(time.perf_counter() - started, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    payload = run(cases, model=args.model, endpoint=args.endpoint, render_only=args.render_only)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm, s in payload["summaries"].items():
        extras = f" success={s['success']:.2f} attribution={s['source_correct']:.2f}" if "success" in s else ""
        print(f"{arm:<15} expected_present={s['expected_present']:.2f} "
              f"cold_kept={s['cold_mappings_present_mean']:.2f}{extras}")


if __name__ == "__main__":
    main()
