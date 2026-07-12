"""Turn a gateway ledger into the receipts report: savings + quality evidence.

    python -m gateway.report results/gateway/ledger.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.brain_runtime.stats import wilson_interval


def summarize(ledger_path: str) -> dict[str, Any]:
    entries = []
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    optimized = [e for e in entries if e.get("optimized")]
    shadows = [e for e in entries if isinstance(e.get("shadow"), dict)]
    chars_saved = sum(
        e.get("request_chars_original", e["request_chars"]) - e["request_chars"]
        for e in optimized
    )
    parity_hits = sum(bool(e["shadow"].get("parity")) for e in shadows)
    attention_receipts = [
        e.get("middleware_receipts", {}).get("AttentionContextMiddleware", {})
        for e in entries
        if "AttentionContextMiddleware" in e.get("middleware_receipts", {})
    ]
    latencies = sorted(float(e.get("latency_ms", 0.0)) for e in entries)
    primary_usage = [e["usage"] for e in entries if isinstance(e.get("usage"), dict)]
    shadow_usage = [e["shadow_usage"] for e in entries if isinstance(e.get("shadow_usage"), dict)]
    paired_usage = [
        (e["usage"], e["shadow_usage"])
        for e in entries
        if isinstance(e.get("usage"), dict) and isinstance(e.get("shadow_usage"), dict)
    ]
    shadow_errors = [e["shadow_error"] for e in entries if isinstance(e.get("shadow_error"), dict)]
    requirement_receipts = [
        e["requirements"] for e in entries if isinstance(e.get("requirements"), dict)
    ]
    shadow_requirement_receipts = [
        e["shadow_requirements"]
        for e in entries
        if isinstance(e.get("shadow_requirements"), dict)
    ]
    paired_requirement_receipts = [
        (e["requirements"], e["shadow_requirements"])
        for e in entries
        if isinstance(e.get("requirements"), dict)
        and isinstance(e.get("shadow_requirements"), dict)
    ]
    requirement_contract_requests = sum(
        int(e.get("requirement_contracts", 0)) > 0 for e in entries
    )
    requirements_checked = sum(
        int(receipt.get("requirements", 0)) for receipt in requirement_receipts
    )
    requirements_passed = sum(
        int(receipt.get("passed", 0)) for receipt in requirement_receipts
    )

    def token_sum(rows: list[dict[str, Any]], field: str) -> int:
        return sum(int(row.get(field, 0)) for row in rows)

    def percentile(fraction: float) -> float:
        if not latencies:
            return 0.0
        index = round((len(latencies) - 1) * fraction)
        return round(latencies[index], 1)

    def counts(field: str) -> dict[str, int]:
        values = Counter(str(e.get(field) or "unknown") for e in entries)
        return dict(sorted(values.items()))

    summary: dict[str, Any] = {
        "requests": len(entries),
        "optimized_requests": len(optimized),
        "cached_requests": sum(bool(e.get("cached")) for e in entries),
        "passthrough_requests": sum(not e.get("optimized") and not e.get("cached") for e in entries),
        "request_chars_saved": chars_saved,
        "mean_latency_ms": (
            round(sum(e.get("latency_ms", 0.0) for e in entries) / len(entries), 1)
            if entries
            else 0.0
        ),
        "p50_latency_ms": percentile(0.50),
        "p95_latency_ms": percentile(0.95),
        "requests_by_path": counts("path"),
        "requests_by_model": counts("model"),
        "responses_by_status": counts("status"),
        "requirement_contract_requests": requirement_contract_requests,
        "requirement_receipts": len(requirement_receipts),
        "requirement_receipt_coverage_rate": (
            len(requirement_receipts) / requirement_contract_requests
            if requirement_contract_requests
            else None
        ),
        "requirements_checked": requirements_checked,
        "requirements_passed": requirements_passed,
        "requirement_retention_rate": (
            requirements_passed / requirements_checked if requirements_checked else None
        ),
        "all_requirements_passed_requests": sum(
            receipt.get("all_passed") is True for receipt in requirement_receipts
        ),
        "shadow_requirement_receipts": len(shadow_requirement_receipts),
        "shadow_requirements_checked": sum(
            int(receipt.get("requirements", 0)) for receipt in shadow_requirement_receipts
        ),
        "shadow_requirements_passed": sum(
            int(receipt.get("passed", 0)) for receipt in shadow_requirement_receipts
        ),
        "paired_requirement_receipts": len(paired_requirement_receipts),
        "measured_requirement_pass_delta": sum(
            int(optimized.get("passed", 0)) - int(raw.get("passed", 0))
            for optimized, raw in paired_requirement_receipts
        ),
        "streamed_requests": sum(bool(e.get("streamed")) for e in entries),
        "incomplete_streams": sum(
            e.get("streamed") is True and e.get("stream_complete") is False for e in entries
        ),
        "upstream_requests": sum(bool(e.get("upstream_called")) for e in entries),
        "usage_receipts": len(primary_usage),
        "usage_coverage_rate": (
            len(primary_usage) / sum(bool(e.get("upstream_called")) for e in entries)
            if any(e.get("upstream_called") for e in entries)
            else None
        ),
        "input_tokens": token_sum(primary_usage, "input_tokens"),
        "output_tokens": token_sum(primary_usage, "output_tokens"),
        "total_tokens": token_sum(primary_usage, "total_tokens"),
        "shadow_input_tokens": token_sum(shadow_usage, "input_tokens"),
        "shadow_output_tokens": token_sum(shadow_usage, "output_tokens"),
        "shadow_total_tokens": token_sum(shadow_usage, "total_tokens"),
        "provider_total_tokens_consumed": (
            token_sum(primary_usage, "total_tokens") + token_sum(shadow_usage, "total_tokens")
        ),
        "paired_usage_receipts": len(paired_usage),
        "measured_input_token_savings": sum(
            int(raw.get("input_tokens", 0)) - int(optimized.get("input_tokens", 0))
            for optimized, raw in paired_usage
        ),
        "attention_route_counts": dict(sorted(Counter(
            str(receipt.get("route") or "not_evaluated") for receipt in attention_receipts
        ).items())),
        "attention_applied_requests": sum(bool(receipt.get("applied")) for receipt in attention_receipts),
        "embedding_cache_hits": sum(int(receipt.get("embedding_cache_hits", 0)) for receipt in attention_receipts),
        "embedding_cache_misses": sum(int(receipt.get("embedding_cache_misses", 0)) for receipt in attention_receipts),
        "shadow_samples": len(shadows),
        "shadow_failures": len(shadow_errors),
        "shadow_failure_types": dict(sorted(Counter(
            str(error.get("type") or "unknown") for error in shadow_errors
        ).items())),
        "quality_parity_rate": (parity_hits / len(shadows)) if shadows else None,
        "quality_parity_ci": (
            [round(v, 4) for v in wilson_interval(parity_hits, len(shadows))] if shadows else None
        ),
        "mean_shadow_similarity": (
            round(sum(e["shadow"].get("similarity", 0.0) for e in shadows) / len(shadows), 4)
            if shadows
            else None
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", help="Path to the gateway JSONL ledger.")
    args = parser.parse_args()
    summary = summarize(args.ledger)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["shadow_samples"]:
        low, high = summary["quality_parity_ci"]
        print(
            f"\nreceipts: {summary['optimized_requests']}/{summary['requests']} requests optimized, "
            f"{summary['request_chars_saved']} request chars saved; quality parity "
            f"{summary['quality_parity_rate']:.1%} (95% CI {low:.1%}-{high:.1%}, "
            f"n={summary['shadow_samples']} shadow samples)"
        )


if __name__ == "__main__":
    main()
