#!/usr/bin/env python3
"""Provider-neutral export/score bridge for frontier context evaluation.

Exported JSONL packets contain complete system/user messages and deterministic
request IDs.  Any frontier provider may execute them; response JSONL is scored
locally with the same objective scorer, keeping provider integration outside the
research contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.context_organization_eval import (
    SYSTEM,
    V2_ARMS,
    prompt_for,
    render_five_arms,
    score_response,
    summarize,
    validate_case,
)
from gateway.context_compiler import ConversationCompiler


def _request_id(case_id: str, arm: str, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{case_id}:{arm}:{digest}"


def export_packets(
    cases: Sequence[Mapping[str, Any]],
    compiler: ConversationCompiler,
    *,
    rewrite_fn: Any | None = None,
) -> list[dict[str, Any]]:
    packets = []
    for raw_case in cases:
        case = validate_case(raw_case)
        contexts, diagnostics = render_five_arms(case, compiler, rewrite_fn=rewrite_fn)
        arms = V2_ARMS if rewrite_fn is not None else tuple(
            arm for arm in V2_ARMS if arm != "llm_rewrite"
        )
        for arm in arms:
            prompt = prompt_for(case, contexts[arm])
            packets.append({
                "request_id": _request_id(str(case["id"]), arm, prompt),
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_output_tokens": 32,
                "metadata": {
                    "benchmark": "deterministic-input-compiler-frontier-v1",
                    "case": case["id"], "arm": arm,
                    "expected": case["expected"],
                    "expected_source": case["expected_source"],
                    "forbidden": case["forbidden"],
                    "context_chars": diagnostics[arm]["context_chars"],
                    "record_ids": diagnostics[arm]["record_ids"],
                    "rewrite_status": diagnostics[arm].get("rewrite_status"),
                },
            })
    return packets


def score_packets(
    packets: Sequence[Mapping[str, Any]], responses: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_id = {str(response["request_id"]): response for response in responses}
    if len(by_id) != len(responses):
        raise ValueError("response request IDs must be unique")
    rows = []
    for packet in packets:
        request_id = str(packet["request_id"])
        if request_id not in by_id:
            raise ValueError(f"missing response: {request_id}")
        response = by_id.pop(request_id)
        text = response.get("text")
        if not isinstance(text, str):
            raise ValueError(f"response text must be a string: {request_id}")
        meta = packet["metadata"]
        case = {
            "expected": meta["expected"],
            "expected_source": meta["expected_source"],
            "forbidden": meta["forbidden"],
        }
        score = score_response(text, case)
        rows.append({
            "request_id": request_id, "case": meta["case"], "arm": meta["arm"],
            "response": text, "context_chars": meta["context_chars"],
            "context_expected_present": str(meta["expected"]) in packet["messages"][1]["content"],
            "context_expected_source_present": str(meta["expected_source"]) in packet["messages"][1]["content"],
            "context_forbidden_present": any(
                str(value) in packet["messages"][1]["content"] for value in meta["forbidden"]
            ),
            "prompt_tokens": int(response.get("prompt_tokens", 0)),
            "completion_tokens": int(response.get("completion_tokens", 0)),
            "total_duration_ns": int(float(response.get("latency_seconds", 0.0)) * 1e9),
            "wall_seconds": float(response.get("latency_seconds", 0.0)),
            **score,
        })
    if by_id:
        raise ValueError(f"unknown response IDs: {sorted(by_id)[:3]}")
    present_arms = [arm for arm in V2_ARMS if any(row["arm"] == arm for row in rows)]
    return {
        "benchmark": "deterministic-input-compiler-frontier-v1",
        "rows": rows,
        "arms": present_arms,
        "summaries": [summarize(rows, arm) for arm in present_arms],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--cases", required=True)
    export.add_argument("--out", required=True)
    score = sub.add_parser("score")
    score.add_argument("--packets", required=True)
    score.add_argument("--responses", required=True)
    score.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "export":
        cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        _write_jsonl(Path(args.out), export_packets(cases, ConversationCompiler()))
        return
    payload = score_packets(_read_jsonl(Path(args.packets)), _read_jsonl(Path(args.responses)))
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
