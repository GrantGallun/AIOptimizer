"""Run the AIOptimizer Gateway.

    python -m aioptimizer --port 8800                       # in front of local Ollama
    python -m aioptimizer --port 8800 --shadow-rate 0.2     # judge 20% of optimized requests

Point any OpenAI-compatible client at http://127.0.0.1:<port>/v1 and read the receipts with
``python -m aioptimizer.report results/gateway/ledger.jsonl``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .cache_middleware import ExactCacheMiddleware
from .config import load_config
from .compact_middleware import CompactContextMiddleware
from .context_middleware import AttentionContextMiddleware
from .ledger import JsonlLedger
from .receipts import ShadowJudge
from .server import GatewayServer


def parse_args(argv=None):
    default_config = os.environ.get("AIOPT_CONFIG")
    if default_config is None and Path("aioptimizer.json").exists():
        default_config = "aioptimizer.json"
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_config)
    known, _ = pre.parse_known_args(argv)
    file_config = load_config(known.config) if known.config else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config, help="Persistent gateway JSON config.")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--upstream", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--upstream-timeout-seconds", type=float, default=300.0,
                        help="Connect/read timeout for each upstream operation.")
    parser.add_argument("--ledger", default="results/gateway/ledger.jsonl")
    parser.add_argument("--budget-chars", type=int, default=12_000, help="Compaction threshold.")
    parser.add_argument("--shadow-rate", type=float, default=0.2,
                        help="Fraction of optimized requests judged against the raw original.")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--no-cache", action="store_true", dest="no_cache")
    cache.add_argument("--cache", action="store_false", dest="no_cache")
    compact = parser.add_mutually_exclusive_group()
    compact.add_argument("--no-compact", action="store_true", dest="no_compact")
    compact.add_argument("--compact", action="store_false", dest="no_compact")
    attention = parser.add_mutually_exclusive_group()
    attention.add_argument(
        "--attention-context",
        action="store_true",
        dest="attention_context",
        help="Enable experimental attention reorganization (default: OFF).",
    )
    attention.add_argument("--no-attention-context", action="store_false", dest="attention_context")
    parser.add_argument("--attention-budget-chars", type=int, default=12_000)
    parser.add_argument("--attention-min-relevance", type=float, default=0.5)
    parser.set_defaults(**file_config)
    return parser.parse_args(argv)


def build_middlewares(args):
    middlewares = []
    if not args.no_cache:
        middlewares.append(ExactCacheMiddleware())
    if args.attention_context:
        middlewares.append(AttentionContextMiddleware(
            budget_chars=args.attention_budget_chars,
            min_relevance=args.attention_min_relevance,
        ))
    if not args.no_compact:
        middlewares.append(CompactContextMiddleware(budget_chars=args.budget_chars))
    return middlewares


def main() -> None:
    args = parse_args()
    middlewares = build_middlewares(args)

    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    server = GatewayServer(
        args.upstream,
        middlewares=tuple(middlewares),
        ledger=JsonlLedger(args.ledger),
        port=args.port,
        shadow=ShadowJudge(rate=args.shadow_rate),
        upstream_timeout=args.upstream_timeout_seconds,
    )
    active = ", ".join(type(m).__name__ for m in middlewares) or "(passthrough)"
    print(f"AIOptimizer Gateway on http://127.0.0.1:{args.port}/v1 -> {args.upstream}")
    print(f"middlewares: {active} | shadow-rate {args.shadow_rate:.0%} | ledger {args.ledger}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
