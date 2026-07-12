"""Persistent, validated JSON configuration for the AIOptimizer gateway."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CONFIG: dict[str, Any] = {
    "port": 8800,
    "upstream": "http://127.0.0.1:11434",
    "ledger": "results/gateway/ledger.jsonl",
    "budget_chars": 12_000,
    "shadow_rate": 0.2,
    "no_cache": False,
    "no_compact": False,
    "attention_context": False,
    "attention_budget_chars": 12_000,
    "attention_min_relevance": 0.5,
}


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(value) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown gateway config keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **dict(value)}
    if not isinstance(config["port"], int) or not 0 <= config["port"] <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    for key in ("budget_chars", "attention_budget_chars"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("shadow_rate", "attention_min_relevance"):
        if not isinstance(config[key], (int, float)) or not 0.0 <= float(config[key]) <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
    for key in ("no_cache", "no_compact", "attention_context"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("upstream", "ledger"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("gateway config must be a JSON object")
    return validate_config(loaded)


def write_default(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check"))
    parser.add_argument("path", nargs="?", default="aioptimizer.json")
    args = parser.parse_args()
    if args.command == "init":
        print(write_default(args.path))
    else:
        print(json.dumps(load_config(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
