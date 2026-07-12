#!/usr/bin/env python3
"""Intentional-ambiguity benchmark for deterministic input compilation.

The active requests deliberately leave direction open.  Cases contain four
independent, source-labelled opportunity signals; scoring rewards grounded
coverage without requiring any particular recommendation or prose style.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.context_organization_eval import (
    V2_ARMS,
    local_rewrite_fn,
    render_five_arms,
)
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient
from gateway.context_compiler import ConversationCompiler

SYSTEM = (
    "The user's request is intentionally open-ended. Preserve that freedom: do not invent a fixed "
    "deliverable, audience, niche, or mandatory recommendation. Discuss whichever directions "
    "genuinely stand out from the supplied context. Cite supporting T#### source turns."
)
QUERIES = (
    "What stands out to you?",
    "Where could this go?",
    "What do you make of this?",
    "Explore what seems promising here.",
)
SIGNALS = (
    ("sig_velora", "Repeated requests show a latency bottleneck that exact caching could remove."),
    ("sig_nembus", "Long conversations lose older decisions, suggesting relevance-organized memory."),
    ("sig_caldor", "Users repeat successful procedures, suggesting reusable workflow capture."),
    ("sig_sorell", "Different models excel on different tasks, suggesting evidence-gated routing."),
    ("sig_talven", "Large source bundles contain irrelevant files, suggesting encoder selection."),
    ("sig_brinex", "Structured outputs fail less often, suggesting schema-aware generation."),
    ("sig_orvane", "Repeated model loading dominates latency, suggesting residency-aware serving."),
    ("sig_delphi", "Prior constraints disappear across sessions, suggesting source-grounded records."),
)
RIGID_PHRASES = (
    "the only goal", "must target", "the required deliverable", "the user requires",
    "the sole direction", "definitely must",
)
CODE_RE = re.compile(r"\bsig_[a-z]{6}\b", re.IGNORECASE)


def make_cases(seed: int, count: int = 12) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases = []
    for case_index in range(count):
        chosen = rng.sample(list(SIGNALS), 4)
        positions = sorted(rng.sample(range(4, 22), 4))
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "Preserve the user's exploratory freedom."}
        ]
        signal_sources = {}
        signal_index = 0
        for turn_index in range(1, 24):
            if signal_index < len(positions) and turn_index == positions[signal_index]:
                code, observation = chosen[signal_index]
                content = f"Observation {code}: {observation}"
                signal_sources[code] = f"T{len(messages) + 1:04d}"
                signal_index += 1
            else:
                content = (
                    f"Routine project note {case_index}-{turn_index}: documentation and tests "
                    "remain under review."
                )
            role = "user" if turn_index % 2 else "assistant"
            messages.append({"role": role, "content": content})
        query = QUERIES[case_index % len(QUERIES)]
        messages.append({"role": "user", "content": query})
        # render overhead is stable; pressure is deliberately moderate so this is
        # a non-inferiority test, not another severe-retention benchmark.
        full_chars = len(json.dumps(messages, ensure_ascii=False))
        cases.append(
            {
                "id": f"ambiguity-{seed}-{case_index:02d}",
                "messages": messages,
                "query": query,
                "signals": [code for code, _ in chosen],
                "signal_sources": signal_sources,
                "budget_chars": int(full_chars * 0.65),
                # Compatibility with the neutral renderer's context diagnostics.
                "expected": chosen[0][0],
                "expected_source": signal_sources[chosen[0][0]],
                "forbidden": [],
            }
        )
    return cases


def score_response(text: str, case: Mapping[str, Any]) -> dict[str, Any]:
    lowered = text.lower()
    mentioned = [code for code in case["signals"] if code in lowered]
    attributed = [
        code for code in mentioned if str(case["signal_sources"][code]).lower() in lowered
    ]
    known_codes = {code for code, _ in SIGNALS}
    code_like = {match.lower() for match in CODE_RE.findall(text)}
    invented = sorted(code_like - known_codes)
    rigid = [phrase for phrase in RIGID_PHRASES if phrase in lowered]
    return {
        "direction_count": len(mentioned),
        "evidence_coverage": len(mentioned) / len(case["signals"]),
        "attributed_directions": len(attributed),
        "source_attribution": len(attributed) / len(mentioned) if mentioned else 0.0,
        "ambiguity_preserved": not rigid,
        "rigid_phrases": rigid,
        "invented_signal_codes": invented,
        "semantic_corruption": bool(invented),
    }


def summarize(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    count = len(selected)
    mean = lambda key: sum(float(row[key]) for row in selected) / count if count else 0.0
    return {
        "arm": arm,
        "cases": count,
        "mean_direction_count": round(mean("direction_count"), 3),
        "mean_evidence_coverage": round(mean("evidence_coverage"), 4),
        "mean_source_attribution": round(mean("source_attribution"), 4),
        "ambiguity_preservation_rate": round(mean("ambiguity_preserved"), 4),
        "semantic_corruption_rate": round(mean("semantic_corruption"), 4),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in selected),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in selected),
        "rewrite_prompt_tokens": sum(int(row["rewrite_prompt_tokens"]) for row in selected),
        "rewrite_completion_tokens": sum(int(row["rewrite_completion_tokens"]) for row in selected),
        "model_duration_seconds": round(
            sum(int(row["total_duration_ns"]) for row in selected) / 1_000_000_000, 3
        ),
        "preprocess_seconds": round(sum(float(row["preprocess_seconds"]) for row in selected), 3),
    }


def run(
    cases: Sequence[Mapping[str, Any]],
    client: OllamaClient,
    *,
    model: str = DEFAULT_MODEL,
    compiler: ConversationCompiler | None = None,
    include_llm_rewrite: bool = True,
) -> dict[str, Any]:
    compiler = compiler or ConversationCompiler()
    rewrite = local_rewrite_fn(client, model) if include_llm_rewrite else None
    rows = []
    started = time.perf_counter()
    for case in cases:
        contexts, diagnostics = render_five_arms(case, compiler, rewrite_fn=rewrite)
        for arm in V2_ARMS:
            generation = client.generate_with_metrics(
                f"Context:\n{contexts[arm]}\n\nUser request (verbatim):\n{case['query']}",
                model=model,
                system=SYSTEM,
                temperature=0.7,
                max_tokens=256,
            )
            rewrite_metrics = diagnostics[arm].get("rewrite_metrics", {})
            rows.append(
                {
                    "case": case["id"], "arm": arm, "response": generation.text,
                    "active_request_verbatim": case["query"] in contexts[arm],
                    "record_ids": diagnostics[arm]["record_ids"],
                    "context_chars": diagnostics[arm]["context_chars"],
                    "preprocess_seconds": diagnostics[arm]["preprocess_seconds"],
                    "rewrite_prompt_tokens": int(rewrite_metrics.get("prompt_tokens", 0)),
                    "rewrite_completion_tokens": int(rewrite_metrics.get("completion_tokens", 0)),
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "total_duration_ns": generation.total_duration_ns,
                    **score_response(generation.text, case),
                }
            )
    return {
        "benchmark": "intentional-ambiguity-v1",
        "seed": None,
        "model": model,
        "arms": list(V2_ARMS),
        "rows": rows,
        "summaries": [summarize(rows, arm) for arm in V2_ARMS],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-llm-rewrite", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    payload = run(
        make_cases(args.seed, args.count),
        OllamaClient(args.endpoint, timeout_seconds=180.0),
        model=args.model,
        include_llm_rewrite=not args.no_llm_rewrite,
    )
    payload["seed"] = args.seed
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
