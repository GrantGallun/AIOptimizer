#!/usr/bin/env python3
"""Replay real Codex session transcripts through the production hook, turn by turn.

    python experiments/brain_runtime/replay_real_sessions.py --top 12

Answers the question a week of organic traffic would answer, today: on REAL work,
how often does context pressure occur, and what does the (fixed) router do about it?

For every user turn of each selected session, the visible history as of that moment
is fed to an in-process `AttentionContextMiddleware` (production defaults) — the same
extractor (`extract_visible_messages`) and the same router as live. The transport
wrapper (`process_hook`, file re-reading, receipt plumbing) is exercised separately by
the e2e delivery tests; replaying THROUGH it re-parses the multi-MB transcript per
turn, which is quadratic and was killed at 120s without finishing one session. Note
one fidelity bound: the extractor's 4MB tail cap is applied to the whole file once,
so for sessions over 4MB the earliest turns replay against an already-windowed
history — the same thing the live hook would have seen only for the LATER turns.
Receipts are content-free; this script reports aggregates only and writes no ledger.

Observational telemetry — no pre-registration, no gate, no hidden discipline. It
describes the pressure distribution of one user's real sessions; it proves nothing
about quality (that is v14/v16/v17/v18's job).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from aioptimizer.codex_hook import MAX_MESSAGES, extract_visible_messages
from aioptimizer.context_middleware import AttentionContextMiddleware

SESSIONS_ROOT = Path.home() / ".codex" / "sessions"
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "brain_runtime"
BUDGET_CHARS = 6_000  # the hook's production default


def replay_session(path: Path,
                   middleware: AttentionContextMiddleware) -> list[dict[str, Any]]:
    visible = extract_visible_messages(path)  # production extractor, parsed ONCE
    receipts = []
    turn_number = 0
    for index, message in enumerate(visible):
        if message["role"] != "user":
            continue
        history = visible[:index][-MAX_MESSAGES:]  # what the live hook would pass on
        prompt = message["content"]
        history_chars = sum(len(m["content"]) for m in history)
        started = time.perf_counter()
        try:
            result = middleware.compile_additional_context(
                history, query=prompt, output_budget_chars=BUDGET_CHARS,
            )
            route = str(result.get("route") or "unknown")
            context = result.get("context") or ""
            receipt = {
                "route": route,
                "route_reason": result.get("route_reason"),
                "injected": route == "attention" and bool(context),
                "output_chars": len(context),
                "error_type": None,
            }
        except Exception as error:  # mirror the hook's fail-open receipt
            receipt = {"route": "error", "route_reason": None, "injected": False,
                       "output_chars": 0, "error_type": type(error).__name__}
        receipts.append({
            "session": path.name,
            "turn": turn_number,
            "history_chars": history_chars,
            "messages": len(history),
            "wall_seconds": round(time.perf_counter() - started, 3),
            **receipt,
        })
        turn_number += 1
    return receipts


def summarize(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    eligible = [r for r in rows if r["history_chars"] > BUDGET_CHARS]
    treated = [r for r in eligible if r["route"] == "attention" and r["injected"]]
    print(f"\nPOOLED: {total} real user turns replayed")
    print(f"  eligible (> {BUDGET_CHARS:,} chars of history): {len(eligible)} "
          f"({len(eligible)/total:.1%} of turns)" if total else "")
    if eligible:
        chars = sorted(r["history_chars"] for r in eligible)
        print(f"  eligible history_chars: median {chars[len(chars)//2]:,} "
              f"max {chars[-1]:,}")
        print(f"  TREATED (route=attention, injected): {len(treated)}/{len(eligible)} "
              f"= {len(treated)/len(eligible):.1%}  <- the number a week of organic "
              f"traffic would have given us")
    print("\n  route mix (all turns):")
    for route, count in Counter(r["route"] for r in rows).most_common():
        print(f"    {route:<18} {count:>5}  ({count/total:.1%})")
    print("  route_reason mix (eligible turns only):")
    for reason, count in Counter(str(r["route_reason"]) for r in eligible).most_common():
        print(f"    {reason:<28} {count:>5}")
    errors = Counter(r["error_type"] for r in rows if r["route"] == "error")
    if errors:
        print(f"  errors: {dict(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=12,
                        help="Replay the N largest session files.")
    parser.add_argument("--out", default=str(OUT_DIR / "real_session_replay_v1.json"))
    args = parser.parse_args()

    sessions = sorted(SESSIONS_ROOT.rglob("*.jsonl"),
                      key=lambda p: p.stat().st_size, reverse=True)[: args.top]
    if not sessions:
        raise SystemExit(f"no session files under {SESSIONS_ROOT}")
    print(f"replaying {len(sessions)} sessions "
          f"({sum(p.stat().st_size for p in sessions)/1e6:.1f} MB of transcript)")

    middleware = AttentionContextMiddleware()  # production defaults, real encoder
    all_rows: list[dict[str, Any]] = []
    for path in sessions:
        started = time.perf_counter()
        rows = replay_session(path, middleware)
        eligible = sum(1 for r in rows if r["history_chars"] > BUDGET_CHARS)
        injected = sum(1 for r in rows if r["injected"])
        print(f"  {path.name[:60]:<62} turns={len(rows):>3} eligible={eligible:>3} "
              f"injected={injected:>3} ({time.perf_counter()-started:.0f}s)", flush=True)
        all_rows.extend(rows)

    out = Path(args.out)
    out.write_text(json.dumps({"schema": "aioptimizer.session-replay.v1",
                               "budget_chars": BUDGET_CHARS,
                               "sessions": len(sessions),
                               "rows": all_rows}, indent=2) + "\n", encoding="utf-8")
    summarize(all_rows)
    print(f"\ncontent-free replay rows -> {out}")


if __name__ == "__main__":
    main()
