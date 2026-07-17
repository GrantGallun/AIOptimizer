"""Treatment integrity: is the optimizer actually being applied where it helps?

    python -m aioptimizer.health .aioptimizer/codex_hook_ledger.jsonl

This exists because every instrument we had said the product was fine while it did
nothing for five days. On 2026-07-17 the live hook ledger showed that of 39 real turns
carrying the exact condition HYP-38 proved we help (history over the injection budget),
the product treated 4 — and the last injection was 2026-07-12. The `episodes inspect`
coverage report said `context_route: rate 1.0` throughout, because **coverage measures
whether a field is present, not whether the treatment fired**. That is the same category
error `evidence.py` names about itself: metadata completeness is not a verdict.

So this module deliberately reports one number that can embarrass us:

    treatment_applied_rate = treated / eligible

`eligible` counts only turns where the treatment could have helped (history over budget).
Turns below the threshold are excluded rather than counted as successes — including them
would let a quiet week of small prompts hide a dead sidecar behind a high rate.

The hook itself is right to fail OPEN: a broken optimizer must never brick a coding
session. Failing open is correct; failing *silent* is the bug. This is the consumer for
the telemetry the hook has been writing all along.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HEALTH_SCHEMA = "aioptimizer.treatment-health.v1"

# The hook's default injection budget. A turn whose history fits in the budget cannot be
# helped by reorganisation, so it is not evidence either way.
DEFAULT_BUDGET_CHARS = 6_000
# Below this, the product is not doing its job even though nothing crashed.
DEFAULT_MIN_RATE = 0.90
# A pressure-carrying turn should be treated regularly; silence for this long is an outage.
DEFAULT_STALE_SECONDS = 24 * 3600


def read_rows(ledger_path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def assess(
    rows: list[dict[str, Any]],
    *,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
    min_rate: float = DEFAULT_MIN_RATE,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    eligible = [
        r for r in rows
        if isinstance(r.get("history_chars"), int) and r["history_chars"] > budget_chars
    ]
    treated = [r for r in eligible if r.get("route") == "attention" and r.get("injected")]
    injections = [r for r in rows if r.get("route") == "attention" and r.get("injected")]
    last_ts = max((r["ts"] for r in injections if isinstance(r.get("ts"), (int, float))), default=None)

    findings: list[str] = []
    rate = (len(treated) / len(eligible)) if eligible else None

    if not eligible:
        findings.append(
            "no_eligible_turns: nothing in this ledger carried enough context to treat; "
            "this reports nothing about product health either way"
        )
    elif rate is not None and rate < min_rate:
        findings.append(
            f"treatment_not_applied: {len(treated)}/{len(eligible)} "
            f"({rate:.1%}) of turns over {budget_chars:,} chars were treated, below {min_rate:.0%}"
        )

    errors = Counter(
        r.get("error_type", "unknown") for r in rows if r.get("route") == "error"
    )
    if errors:
        top = ", ".join(f"{name} x{n}" for name, n in errors.most_common(3))
        findings.append(f"optimizer_errors: {sum(errors.values())} turn(s) failed open ({top})")
        if any("URLError" in str(name) or "Timeout" in str(name) for name in errors):
            findings.append(
                "sidecar_unreachable: URLError/Timeout means the hook could not reach the "
                "optimizer at all — is the sidecar running?"
            )

    if last_ts is None and eligible:
        findings.append("never_injected: this ledger contains no successful injection")
    elif last_ts is not None and (now - last_ts) > stale_seconds:
        findings.append(
            f"stale_injection: last successful injection was "
            f"{(now - last_ts) / 3600:.1f}h ago (over {stale_seconds / 3600:.0f}h)"
        )

    unexplained = [r for r in eligible if r.get("route") == "raw" and not r.get("route_reason")]
    if unexplained:
        findings.append(
            f"unexplained_bypass: {len(unexplained)} eligible turn(s) routed 'raw' with no "
            "route_reason — the bypass cannot be diagnosed from this ledger"
        )

    return {
        "schema": HEALTH_SCHEMA,
        "turns": len(rows),
        "eligible_turns": len(eligible),
        "treated_turns": len(treated),
        "treatment_applied_rate": rate,
        "budget_chars": budget_chars,
        "last_injection_ts": last_ts,
        "last_injection_age_hours": None if last_ts is None else round((now - last_ts) / 3600, 2),
        "route_mix": dict(Counter(r.get("route", "unknown") for r in rows)),
        "error_types": dict(errors),
        "findings": findings,
        "healthy": not findings,
    }


def format_report(report: dict[str, Any]) -> str:
    rate = report["treatment_applied_rate"]
    rate_text = "n/a" if rate is None else f"{rate:.1%}"
    lines = [
        "AIOptimizer — treatment integrity",
        f"  turns recorded         {report['turns']}",
        f"  eligible (> {report['budget_chars']:,} chars)  {report['eligible_turns']}",
        f"  treated                {report['treated_turns']}",
        f"  TREATMENT APPLIED RATE {rate_text}",
        f"  last injection         "
        + ("never" if report["last_injection_age_hours"] is None
           else f"{report['last_injection_age_hours']}h ago"),
        f"  route mix              {report['route_mix']}",
    ]
    if report["healthy"]:
        lines.append("\n  OK — the optimizer is being applied where it can help.")
    else:
        lines.append("\n  PROBLEMS:")
        lines.extend(f"    - {f}" for f in report["findings"])
    return "\n".join(lines)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", help="Hook ledger JSONL (e.g. .aioptimizer/codex_hook_ledger.jsonl).")
    parser.add_argument("--budget-chars", type=int, default=DEFAULT_BUDGET_CHARS)
    parser.add_argument("--min-rate", type=float, default=DEFAULT_MIN_RATE)
    parser.add_argument("--stale-hours", type=float, default=DEFAULT_STALE_SECONDS / 3600)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = assess(
        read_rows(args.ledger),
        budget_chars=args.budget_chars,
        min_rate=args.min_rate,
        stale_seconds=args.stale_hours * 3600,
    )
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    # Non-zero exit so this can gate a usefulness claim, or a CI job, rather than being
    # one more report nobody reads.
    raise SystemExit(0 if report["healthy"] else 1)


if __name__ == "__main__":
    main()
