"""Public API for the local OpenAI-compatible optimization gateway."""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AttentionContextMiddleware",
    "CompactContextMiddleware",
    "ConversationCompiler",
    "ExactCacheMiddleware",
    "GatewayServer",
    "JsonlLedger",
    "Middleware",
    "PassthroughMiddleware",
    "SemanticCacheMiddleware",
    "ShadowJudge",
    "ShortCircuit",
]

_EXPORTS = {
    "AttentionContextMiddleware": ("context_middleware", "AttentionContextMiddleware"),
    "CompactContextMiddleware": ("compact_middleware", "CompactContextMiddleware"),
    "ConversationCompiler": ("context_compiler", "ConversationCompiler"),
    "ExactCacheMiddleware": ("cache_middleware", "ExactCacheMiddleware"),
    "GatewayServer": ("server", "GatewayServer"),
    "JsonlLedger": ("ledger", "JsonlLedger"),
    "Middleware": ("middleware", "Middleware"),
    "PassthroughMiddleware": ("middleware", "PassthroughMiddleware"),
    "SemanticCacheMiddleware": ("semantic_cache", "SemanticCacheMiddleware"),
    "ShadowJudge": ("receipts", "ShadowJudge"),
    "ShortCircuit": ("middleware", "ShortCircuit"),
}


def __getattr__(name: str):
    """Load public classes lazily so the research compatibility layer cannot cycle."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
