"""Turn a gateway ledger into the receipts report: savings + quality evidence.

    python -m gateway.report results/gateway/ledger.jsonl
"""

from __future__ import annotations

import argparse
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
    summary: dict[str, Any] = {
        "requests": len(entries),
        "optimized_requests": len(optimized),
        "request_chars_saved": chars_saved,
        "mean_latency_ms": (
            round(sum(e.get("latency_ms", 0.0) for e in entries) / len(entries), 1)
            if entries
            else 0.0
        ),
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
