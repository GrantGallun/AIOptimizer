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
        "attention_route_counts": dict(sorted(Counter(
            str(receipt.get("route") or "not_evaluated") for receipt in attention_receipts
        ).items())),
        "attention_applied_requests": sum(bool(receipt.get("applied")) for receipt in attention_receipts),
        "embedding_cache_hits": sum(int(receipt.get("embedding_cache_hits", 0)) for receipt in attention_receipts),
        "embedding_cache_misses": sum(int(receipt.get("embedding_cache_misses", 0)) for receipt in attention_receipts),
        "shadow_samples": len(shadows),
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
