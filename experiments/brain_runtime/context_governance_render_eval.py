#!/usr/bin/env python3
"""v14 candidate: typed governance internal, attention presentation unchanged."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.context_organization_eval import (
    SYSTEM, prompt_for, score_response, summarize, validate_case,
)
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient
from gateway.context_compiler import ConversationCompiler

ARMS = ("attention", "governed_attention")


def render_arms(case: Mapping[str, Any], compiler: ConversationCompiler):
    case = validate_case(case)
    compiled = compiler.compile(case["messages"])
    pinned = compiler.pinned_records(compiled)
    pinned_ids = {record["id"] for record in pinned}
    ranked = compiler.materialize(
        compiled, query=str(case["query"]), max_records=len(compiled.records)
    )["working_set"]
    candidates = [record for record in ranked if record["id"] not in pinned_ids]
    attention, _, selected = compiler.matched_record_pair(
        pinned, candidates, budget_chars=int(case["budget_chars"])
    )
    governed = compiler.render_governed_record_set(compiled, selected)
    if attention != governed:
        raise AssertionError("governed presentation must be byte-identical to attention")
    return {
        "attention": attention,
        "governed_attention": governed,
    }, {
        "compiler_fingerprint": compiler.fingerprint(compiled),
        "record_ids": [record["id"] for record in selected],
        "byte_identical": True,
        "integrity": compiler.audit_integrity(compiled),
    }


def run(
    cases: Sequence[Mapping[str, Any]], client: OllamaClient, *,
    model: str = DEFAULT_MODEL, compiler: ConversationCompiler | None = None,
) -> dict[str, Any]:
    compiler = compiler or ConversationCompiler()
    rows = []; started = time.perf_counter()
    for raw_case in cases:
        case = validate_case(raw_case)
        contexts, diagnostics = render_arms(case, compiler)
        for arm in ARMS:
            generation = client.generate_with_metrics(
                prompt_for(case, contexts[arm]), model=model, system=SYSTEM,
                temperature=0.0, max_tokens=32,
            )
            rows.append({
                "case": case["id"], "arm": arm, "response": generation.text,
                "context_chars": len(contexts[arm]),
                "context_expected_present": str(case["expected"]) in contexts[arm],
                "context_expected_source_present": str(case["expected_source"]) in contexts[arm],
                "context_forbidden_present": any(
                    str(value) in contexts[arm] for value in case.get("forbidden", [])
                ),
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_duration_ns": generation.total_duration_ns,
                "load_duration_ns": generation.load_duration_ns,
                "wall_seconds": generation.total_duration_ns / 1e9,
                "compiler_fingerprint": diagnostics["compiler_fingerprint"],
                "record_ids": diagnostics["record_ids"],
                "byte_identical": diagnostics["byte_identical"],
                **score_response(generation.text, case),
            })
    return {
        "benchmark": "context-governance-render-v14-candidate",
        "model": model, "arms": list(ARMS), "rows": rows,
        "summaries": [summarize(rows, arm) for arm in ARMS],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
