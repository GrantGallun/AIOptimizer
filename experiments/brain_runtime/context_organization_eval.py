#!/usr/bin/env python3
"""Evaluate raw chronology versus local-attention context organization.

Cases are supplied as an external frozen JSON file so this implementation does
not choose research seeds or gates.  Both arms share the same source transcript,
character budget, model settings, system instruction, and deterministic scorer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient
from gateway.context_compiler import ConversationCompiler

ARMS = ("raw", "attention")
ANSWER_RE = re.compile(r"^\s*([^|\r\n]+)\|\s*(T\d{4})\s*$")
SYSTEM = (
    "Use only the supplied conversation context. Reply with the actual answer, a pipe, and the "
    "supporting T#### turn ID; for example: blue|T0007. Do not output labels or R#### record IDs. "
    "The source turn must STATE the answer; never cite the turn that asks the current question. "
    "Use the T#### value shown after 'source:' in that fact's context header."
)


def validate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    required = ("id", "messages", "query", "expected", "expected_source", "budget_chars")
    if any(key not in case for key in required):
        raise ValueError(f"case requires: {', '.join(required)}")
    if not isinstance(case["messages"], list) or not case["messages"]:
        raise ValueError("case messages must be a non-empty list")
    if not isinstance(case["budget_chars"], int) or case["budget_chars"] <= 0:
        raise ValueError("case budget_chars must be positive")
    normalized = dict(case)
    normalized["forbidden"] = [str(value) for value in case.get("forbidden", [])]
    return normalized


def render_arms(
    case: Mapping[str, Any], compiler: ConversationCompiler
) -> tuple[dict[str, str], dict[str, Any]]:
    case = validate_case(case)
    compiled = compiler.compile(case["messages"])
    organized = compiler.organize(
        compiled,
        query=str(case["query"]),
        max_records=len(compiled.records),
    )
    budget = case["budget_chars"]
    contexts = {
        "raw": compiler.render_raw(compiled, budget_chars=budget),
        "attention": compiler.render_organized(organized, budget_chars=budget),
    }
    diagnostics = {
        arm: {
            "context_chars": len(context),
            "expected_present": str(case["expected"]) in context,
            "expected_source_present": str(case["expected_source"]) in context,
            "forbidden_present": any(value in context for value in case["forbidden"]),
        }
        for arm, context in contexts.items()
    }
    return contexts, diagnostics


def prompt_for(case: Mapping[str, Any], context: str) -> str:
    return f"Question: {case['query']}\n\nCompiled conversation context:\n{context}\n\nAnswer:"


def score_response(text: str, case: Mapping[str, Any]) -> dict[str, Any]:
    match = ANSWER_RE.fullmatch(text)
    value = match.group(1).strip() if match else None
    source = match.group(2) if match else None
    expected = str(case["expected"]).strip()
    forbidden = [str(item) for item in case.get("forbidden", [])]
    leaked = any(item and item.lower() in text.lower() for item in forbidden)
    return {
        "instruction_retained": match is not None,
        "answer_correct": value == expected and not leaked,
        "source_correct": source == str(case["expected_source"]),
        "success": value == expected and source == str(case["expected_source"]) and not leaked,
        "corruption_or_leak": leaked,
        "parsed_value": value,
        "parsed_source": source,
    }


def summarize(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    count = len(selected)
    rate = lambda key: sum(bool(row[key]) for row in selected) / count if count else 0.0
    return {
        "arm": arm,
        "cases": count,
        "accuracy": rate("answer_correct"),
        "instruction_retention": rate("instruction_retained"),
        "source_attribution": rate("source_correct"),
        "end_to_end_success": rate("success"),
        "context_fact_retention": rate("context_expected_present"),
        "context_source_retention": rate("context_expected_source_present"),
        "context_information_loss": (
            sum(
                not row["context_expected_present"] or not row["context_expected_source_present"]
                for row in selected
            ) / count if count else 0.0
        ),
        "context_forbidden_exposure": rate("context_forbidden_present"),
        "corruption_or_leak_rate": rate("corruption_or_leak"),
        "context_chars": sum(int(row["context_chars"]) for row in selected),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in selected),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in selected),
        "model_duration_seconds": round(
            sum(int(row["total_duration_ns"]) for row in selected) / 1_000_000_000, 3
        ),
        "wall_seconds": round(sum(float(row["wall_seconds"]) for row in selected), 3),
    }


def run_evaluation(
    cases: Sequence[Mapping[str, Any]],
    client: OllamaClient,
    *,
    model: str = DEFAULT_MODEL,
    compiler: ConversationCompiler | None = None,
) -> dict[str, Any]:
    compiler = compiler or ConversationCompiler()
    rows = []
    started = time.perf_counter()
    for raw_case in cases:
        case = validate_case(raw_case)
        contexts, diagnostics = render_arms(case, compiler)
        for arm in ARMS:
            wall_started = time.perf_counter()
            generation = client.generate_with_metrics(
                prompt_for(case, contexts[arm]),
                model=model,
                system=SYSTEM,
                temperature=0.0,
                max_tokens=32,
            )
            wall_seconds = time.perf_counter() - wall_started
            score = score_response(generation.text, case)
            rows.append(
                {
                    "case": case["id"],
                    "arm": arm,
                    "expected": case["expected"],
                    "expected_source": case["expected_source"],
                    "response": generation.text,
                    "context_chars": diagnostics[arm]["context_chars"],
                    "context_expected_present": diagnostics[arm]["expected_present"],
                    "context_expected_source_present": diagnostics[arm]["expected_source_present"],
                    "context_forbidden_present": diagnostics[arm]["forbidden_present"],
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "total_duration_ns": generation.total_duration_ns,
                    "load_duration_ns": generation.load_duration_ns,
                    "wall_seconds": round(wall_seconds, 6),
                    **score,
                }
            )
    return {
        "benchmark": "attention-context-organization-v1",
        "model": model,
        "arms": list(ARMS),
        "rows": rows,
        "summaries": [summarize(rows, arm) for arm in ARMS],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Frozen JSON list of evaluation cases.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--out", required=True, help="New versioned result JSON path.")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("cases file must contain a JSON list")
    if args.render_only:
        compiler = ConversationCompiler()
        payload = [
            {"case": case["id"], "contexts": render_arms(case, compiler)[0]}
            for case in cases
        ]
    else:
        payload = run_evaluation(
            cases,
            OllamaClient(args.endpoint, timeout_seconds=180.0),
            model=args.model,
        )
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"result already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if isinstance(payload, dict):
        print(json.dumps(payload["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
