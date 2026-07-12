"""Deprecated compatibility import for :mod:`aioptimizer`."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "the gateway package is deprecated; import aioptimizer instead",
    DeprecationWarning,
    stacklevel=2,
)

from aioptimizer import *  # noqa: F401,F403
from aioptimizer import __all__, __version__

_MODULES = (
    "cache_middleware",
    "compact_middleware",
    "config",
    "context_compiler",
    "context_middleware",
    "encoder",
    "ledger",
    "middleware",
    "receipts",
    "report",
    "requirements",
    "semantic_cache",
    "server",
    "stats",
    "usage",
)
for _name in _MODULES:
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"aioptimizer.{_name}")

del _MODULES, _name, importlib, sys, warnings
