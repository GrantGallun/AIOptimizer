"""Run the AIOptimizer Gateway.

    python -m gateway --port 8800                       # in front of local Ollama
    python -m gateway --port 8800 --shadow-rate 0.2     # judge 20% of optimized requests

Point any OpenAI-compatible client at http://127.0.0.1:<port>/v1 and read the receipts with
``python -m gateway.report results/gateway/ledger.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from gateway.cache_middleware import ExactCacheMiddleware
from gateway.compact_middleware import CompactContextMiddleware
from gateway.context_middleware import AttentionContextMiddleware
from gateway.ledger import JsonlLedger
from gateway.receipts import ShadowJudge
from gateway.server import GatewayServer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--ledger", default="results/gateway/ledger.jsonl")
    parser.add_argument("--budget-chars", type=int, default=12_000, help="Compaction threshold.")
    parser.add_argument("--shadow-rate", type=float, default=0.2,
                        help="Fraction of optimized requests judged against the raw original.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-compact", action="store_true")
    parser.add_argument(
        "--attention-context",
        action="store_true",
        help="Enable experimental attention reorganization (default: OFF).",
    )
    parser.add_argument("--attention-budget-chars", type=int, default=12_000)
    parser.add_argument("--attention-min-relevance", type=float, default=0.5)
    args = parser.parse_args()

    middlewares = []
    if not args.no_cache:
        middlewares.append(ExactCacheMiddleware())
    if not args.no_compact:
        middlewares.append(CompactContextMiddleware(budget_chars=args.budget_chars))
    if args.attention_context:
        middlewares.append(AttentionContextMiddleware(
            budget_chars=args.attention_budget_chars,
            min_relevance=args.attention_min_relevance,
        ))

    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    server = GatewayServer(
        args.upstream,
        middlewares=tuple(middlewares),
        ledger=JsonlLedger(args.ledger),
        port=args.port,
        shadow=ShadowJudge(rate=args.shadow_rate),
    )
    active = ", ".join(type(m).__name__ for m in middlewares) or "(passthrough)"
    print(f"AIOptimizer Gateway on http://127.0.0.1:{args.port}/v1 -> {args.upstream}")
    print(f"middlewares: {active} | shadow-rate {args.shadow_rate:.0%} | ledger {args.ledger}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
