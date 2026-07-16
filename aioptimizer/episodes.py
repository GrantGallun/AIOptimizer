"""Content-free episode receipts and immutable replay datasets.

The live ledgers are append-only event streams.  This module joins events by
opaque episode id, derives outcome and hard-case labels, freezes a deterministic
development/hidden split, and writes a versioned artifact exactly once.  It
never stores prompt, response, acceptance-command, or producer-result text.

    python -m aioptimizer.episodes build --events hook.jsonl --events bus.jsonl \
        --out results/episodes/episode_dataset_v1.json --split-salt frozen-v1
    python -m aioptimizer.episodes inspect --workspace .
    python -m aioptimizer.episodes hard-cases --dataset ... --out ...
    python -m aioptimizer.episodes replay --dataset ... --split dev
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable
import uuid


EVENT_SCHEMA = "aioptimizer.episode-event.v1"
DATASET_SCHEMA = "aioptimizer.episode-dataset.v1"
HARD_CASE_SCHEMA = "aioptimizer.hard-cases.v1"
HEALTH_SCHEMA = "aioptimizer.episode-health.v1"
STANDARD_EVENT_RELATIVE_PATHS = (
    Path(".aioptimizer/codex_hook_ledger.jsonl"),
    Path(".aioptimizer/hook_ledger.jsonl"),
    Path(".aioptimizer/gateway_ledger.jsonl"),
    Path("agent_bus/episode_events.jsonl"),
    Path("results/gateway/ledger.jsonl"),
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{7,79}$")
_CATEGORY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,79}$")

# Strings in an episode event must be bounded categorical labels.  Numeric and
# boolean measurements may use any non-sensitive key, while these names are the
# only fields allowed to carry strings.
_CATEGORY_KEYS = frozenset({
    "error_type", "operation", "provider", "route", "route_reason", "status",
    "sidecar_error_type", "sidecar_state", "tier", "verifier", "workspace_state",
})
_FORBIDDEN_KEY_PARTS = (
    "acceptance", "argument", "body", "command", "content", "message_text",
    "output_text", "producer_result", "prompt", "query", "response_text",
    "secret", "spec", "title", "transcript",
)


def episode_id_from_payload(payload: dict[str, Any]) -> str:
    """Return a stable opaque id for one hook session without retaining its text."""
    source = payload.get("session_id")
    if not isinstance(source, str) or not source:
        source = payload.get("transcript_path")
    if not isinstance(source, str) or not source:
        return "ep-" + uuid.uuid4().hex[:24]
    digest = hashlib.sha256(
        b"aioptimizer-episode-v1\0" + source.encode("utf-8", errors="replace")
    ).hexdigest()[:24]
    return "ep-" + digest


def new_turn_id() -> str:
    """Return an opaque id unique to one hook/gateway turn."""
    return "turn-" + uuid.uuid4().hex[:24]


def validate_event_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"{field} must be an opaque 8-80 character identifier")
    return value


def content_free_measurements(values: dict[str, Any]) -> dict[str, Any]:
    """Validate and copy bounded scalar measurements safe for durable learning."""
    clean: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key or any(part in key.lower() for part in _FORBIDDEN_KEY_PARTS):
            raise ValueError(f"measurement key is not content-free: {key!r}")
        if value is None:
            continue
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            clean[key] = value
        elif key in _CATEGORY_KEYS and isinstance(value, str) and _CATEGORY_RE.fullmatch(value):
            clean[key] = value
        else:
            raise ValueError(f"measurement {key!r} must be numeric, boolean, or a bounded category")
    return clean


def context_result_measurements(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten a compiler result into the approved content-free feature set."""
    measurements: dict[str, Any] = {}
    for key in (
        "route", "route_reason", "history_chars", "output_chars", "integrity_ok",
        "fail_open", "source_records", "embedding_cache_hits", "embedding_cache_misses",
    ):
        if key in result:
            measurements[key] = result[key]
    load = result.get("load")
    if isinstance(load, dict):
        for key in (
            "token_like_count", "word_token_count", "unique_token_ratio",
            "compression_ratio", "duplicate_turn_ratio", "max_token_run",
            "max_char_run", "repeated_run_ratio", "turn_count", "candidate_count",
            "recent_candidate_count", "obscured_record_count", "repetitive",
            "load_pressure",
        ):
            if key in load:
                measurements[f"load_{key}"] = load[key]
    relevance = result.get("relevance")
    if isinstance(relevance, dict):
        for key in (
            "candidates", "rankable_candidates", "peak", "margin", "mean",
            "best_record_age", "best_record_age_records", "best_in_recent_tail",
            "recent_candidates", "recent_peak",
        ):
            if key in relevance:
                measurements[f"relevance_{key}"] = relevance[key]
    return content_free_measurements(measurements)


def make_event(
    *,
    source: str,
    event_type: str,
    episode_id: str,
    turn_id: str | None = None,
    ts: float | None = None,
    event_id: str | None = None,
    **measurements: Any,
) -> dict[str, Any]:
    """Build one validated content-free event row."""
    if not isinstance(source, str) or not _CATEGORY_RE.fullmatch(source):
        raise ValueError("source must be a bounded category")
    if not isinstance(event_type, str) or not _CATEGORY_RE.fullmatch(event_type):
        raise ValueError("event_type must be a bounded category")
    row: dict[str, Any] = {
        "schema": EVENT_SCHEMA,
        "event_id": validate_event_id(event_id or "evt-" + uuid.uuid4().hex, "event_id"),
        "episode_id": validate_event_id(episode_id, "episode_id"),
        "source": source,
        "event_type": event_type,
        "ts": float(time.time() if ts is None else ts),
    }
    if not math.isfinite(row["ts"]):
        raise ValueError("ts must be finite")
    if turn_id is not None:
        row["turn_id"] = validate_event_id(turn_id, "turn_id")
    row.update(content_free_measurements(measurements))
    return row


class EpisodeEventLedger:
    """Cross-process append-only JSONL sink for validated episode events."""

    def __init__(self, path: str | Path, *, source: str) -> None:
        if not isinstance(source, str) or not _CATEGORY_RE.fullmatch(source):
            raise ValueError("source must be a bounded category")
        self.path = Path(path)
        self.source = source

    def record(
        self,
        event_type: str,
        *,
        episode_id: str,
        turn_id: str | None = None,
        **measurements: Any,
    ) -> dict[str, Any]:
        row = make_event(
            source=self.source,
            event_type=event_type,
            episode_id=episode_id,
            turn_id=turn_id,
            **measurements,
        )
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            if os.write(descriptor, payload) != len(payload):
                raise OSError("episode event append was incomplete")
        finally:
            os.close(descriptor)
        return row


def read_events(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read episode events, ignoring unrelated legacy rows and failing unsafe rows closed."""
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                raw = json.loads(line)
                if isinstance(raw, dict) and raw.get("schema") != EVENT_SCHEMA:
                    unsafe_keys = [
                        key for key in raw
                        if isinstance(key, str)
                        and key != "query_sha256_16"
                        and any(part in key.lower() for part in _FORBIDDEN_KEY_PARTS)
                    ]
                    if unsafe_keys:
                        raise ValueError(
                            f"non-event row contains content-bearing keys: {unsafe_keys}"
                        )
                    continue
                event = _validated_existing_event(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid episode event at {path.name}:{line_number}: {error}") from error
            if event["event_id"] in seen:
                continue
            seen.add(event["event_id"])
            events.append(event)
    return sorted(events, key=lambda item: (item["ts"], item["event_id"]))


def discover_event_paths(
    workspace: str | Path = ".",
    *,
    extra_paths: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Find existing standard event streams without creating runtime files.

    Relative extra paths are resolved against ``workspace``.  Paths are
    returned in a stable order and de-duplicated, which makes the same command
    usable from an AIOptimizer checkout or an unrelated plugin workspace.
    """
    root = Path(workspace)
    candidates = [root / relative for relative in STANDARD_EVENT_RELATIVE_PATHS]
    candidates.extend(
        path if path.is_absolute() else root / path
        for path in (Path(value) for value in extra_paths)
    )
    discovered: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate.resolve(strict=False)))
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        discovered.append(candidate)
    return tuple(discovered)


def inspect_event_stream(event_paths: Iterable[str | Path]) -> dict[str, Any]:
    """Summarize join and outcome coverage without returning episode IDs or text."""
    paths = [Path(path) for path in event_paths]
    events = read_events(paths)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(event["episode_id"], []).append(event)

    def episode_count(predicate) -> int:
        return sum(bool(predicate(rows)) for rows in grouped.values())

    def has_any(rows: list[dict[str, Any]], *keys: str) -> bool:
        return any(any(key in event for key in keys) for event in rows)

    episode_total = len(grouped)
    coverage_counts = {
        "context_route": episode_count(
            lambda rows: any(
                event["event_type"] in {"context_route", "context_compiled"}
                for event in rows
            )
        ),
        "provider_response": episode_count(
            lambda rows: any(event["event_type"] == "provider_response" for event in rows)
        ),
        "provider_usage": episode_count(
            lambda rows: has_any(
                rows, "input_tokens", "output_tokens", "total_tokens",
                "cached_input_tokens", "cache_read_input_tokens",
            )
        ),
        "latency": episode_count(lambda rows: has_any(rows, "latency_ms", "cost_seconds")),
        "acceptance_test": episode_count(lambda rows: has_any(rows, "test_executed", "test_ok")),
        "verification": episode_count(lambda rows: has_any(rows, "verification_ok")),
        "retry": episode_count(lambda rows: has_any(rows, "retry_scheduled")),
        "requirements": episode_count(lambda rows: has_any(rows, "requirements_all_passed")),
        "eventual_outcome": episode_count(lambda rows: has_any(rows, "eventual_success")),
        "cross_source_join": episode_count(
            lambda rows: len({event["source"] for event in rows}) > 1
        ),
    }
    eventual_values = [
        _last_value(rows, "eventual_success") for rows in grouped.values()
        if has_any(rows, "eventual_success")
    ]
    warnings: list[str] = []
    if not paths:
        warnings.append("no_event_files")
    elif not events:
        warnings.append("no_schema_events")
    if episode_total and coverage_counts["cross_source_join"] == 0:
        warnings.append("no_cross_source_joins")
    if episode_total and coverage_counts["eventual_outcome"] == 0:
        warnings.append("no_outcome_labels")
    if episode_total and coverage_counts["provider_usage"] == 0:
        warnings.append("no_provider_usage")

    def coverage_row(count: int) -> dict[str, Any]:
        return {
            "episodes": count,
            "rate": round(count / episode_total, 6) if episode_total else None,
        }

    return {
        "schema": HEALTH_SCHEMA,
        "event_file_count": len(paths),
        "event_count": len(events),
        "episode_count": episode_total,
        "turn_count": len({event["turn_id"] for event in events if "turn_id" in event}),
        "hard_episode_count": episode_count(lambda rows: bool(_hard_reason_codes(rows))),
        "eventual_success_episodes": sum(value is True for value in eventual_values),
        "eventual_failure_episodes": sum(value is False for value in eventual_values),
        "sources": dict(sorted(Counter(event["source"] for event in events).items())),
        "event_types": dict(sorted(Counter(event["event_type"] for event in events).items())),
        "routes": dict(sorted(Counter(
            str(event["route"]) for event in events if "route" in event
        ).items())),
        "coverage": {
            name: coverage_row(count) for name, count in coverage_counts.items()
        },
        "warnings": warnings,
        "note": "Coverage diagnostics only; no controller or research verdict is produced.",
    }


def inspect_workspace(
    workspace: str | Path = ".",
    *,
    extra_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Discover and inspect all standard content-free streams in one workspace."""
    return inspect_event_stream(discover_event_paths(workspace, extra_paths=extra_paths))


def _validated_existing_event(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("schema") != EVENT_SCHEMA:
        raise ValueError(f"schema must be {EVENT_SCHEMA}")
    reserved = {"schema", "event_id", "episode_id", "turn_id", "source", "event_type", "ts"}
    return make_event(
        source=raw.get("source"),
        event_type=raw.get("event_type"),
        episode_id=raw.get("episode_id"),
        turn_id=raw.get("turn_id"),
        ts=raw.get("ts"),
        event_id=raw.get("event_id"),
        **{key: value for key, value in raw.items() if key not in reserved},
    )


def build_dataset(
    event_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    split_salt: str,
    hidden_fraction: float = 0.2,
) -> dict[str, Any]:
    """Join events into an immutable, deterministically split replay artifact."""
    if not isinstance(split_salt, str) or not split_salt:
        raise ValueError("split_salt must be non-empty")
    if not isinstance(hidden_fraction, (int, float)) or isinstance(hidden_fraction, bool) or not 0 < hidden_fraction < 1:
        raise ValueError("hidden_fraction must be between 0 and 1")
    paths = [Path(path) for path in event_paths]
    events = read_events(paths)
    if not events:
        raise ValueError("at least one schema-valid episode event is required")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(event["episode_id"], []).append(event)
    episodes = [
        _summarize_episode(episode_id, rows, split_salt, float(hidden_fraction))
        for episode_id, rows in sorted(grouped.items())
    ]
    source_manifest = [
        {
            "name": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ]
    encoded_events = "\n".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) for event in events
    ).encode("utf-8")
    artifact = {
        "schema": DATASET_SCHEMA,
        "created_at": time.time(),
        "source_manifest": source_manifest,
        "event_stream_sha256": hashlib.sha256(encoded_events).hexdigest(),
        "split": {
            "method": "sha256-threshold-v1",
            "salt_sha256": hashlib.sha256(split_salt.encode("utf-8")).hexdigest(),
            "hidden_fraction": float(hidden_fraction),
        },
        "episode_count": len(episodes),
        "event_count": len(events),
        "hard_episode_count": sum(bool(row["hard_reason_codes"]) for row in episodes),
        "split_counts": dict(sorted(Counter(row["split"] for row in episodes).items())),
        "episodes": episodes,
        "note": "Plumbing artifact only; labels and split membership are not a research verdict.",
    }
    _write_immutable_json(output_path, artifact)
    return artifact


def _summarize_episode(
    episode_id: str,
    events: list[dict[str, Any]],
    split_salt: str,
    hidden_fraction: float,
) -> dict[str, Any]:
    hard_reasons = _hard_reason_codes(events)

    threshold = int(hidden_fraction * (2**64))
    bucket = int.from_bytes(
        hashlib.sha256(f"{split_salt}\0{episode_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    turns = {event["turn_id"] for event in events if "turn_id" in event}
    return {
        "episode_id": episode_id,
        "split": "hidden" if bucket < threshold else "dev",
        "event_count": len(events),
        "turn_count": len(turns),
        "hard_reason_codes": sorted(hard_reasons),
        "eventual_success": _last_value(events, "eventual_success"),
        "events": events,
    }


def _hard_reason_codes(events: list[dict[str, Any]]) -> set[str]:
    hard_reasons: set[str] = set()
    for event in events:
        route = event.get("route")
        if route in {"error", "invalid_input", "sidecar_error"}:
            hard_reasons.add("route_failure")
        status = event.get("status")
        if isinstance(status, int) and status >= 400:
            hard_reasons.add("provider_failure")
        if event.get("producer_ok") is False:
            hard_reasons.add("producer_failure")
        if event.get("verification_ok") is False:
            hard_reasons.add("verification_failure")
        if event.get("retry_scheduled") is True:
            hard_reasons.add("retry_required")
        if event.get("requirements_all_passed") is False:
            hard_reasons.add("requirement_failure")
        if event.get("eventual_success") is False:
            hard_reasons.add("terminal_failure")
    return hard_reasons


def _last_value(events: list[dict[str, Any]], key: str) -> Any:
    values = [event[key] for event in events if key in event]
    return values[-1] if values else None


def build_hard_cases(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    split: str = "dev",
) -> dict[str, Any]:
    """Write an immutable replay view containing only hard episodes in one split."""
    dataset = _load_dataset(dataset_path)
    rows = [
        row for row in dataset["episodes"]
        if row.get("split") == split and row.get("hard_reason_codes")
    ]
    artifact = {
        "schema": HARD_CASE_SCHEMA,
        "created_at": time.time(),
        "dataset_name": Path(dataset_path).name,
        "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
        "split": split,
        "episode_count": len(rows),
        "episodes": rows,
        "note": "Hard-case selection is mechanical and carries no research verdict.",
    }
    _write_immutable_json(output_path, artifact)
    return artifact


def replay_rows(
    dataset_path: str | Path,
    *,
    split: str,
    hard_only: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return frozen rows for a caller-controlled dev or hidden replay."""
    if split not in {"dev", "hidden"}:
        raise ValueError("split must be dev or hidden")
    dataset = _load_dataset(dataset_path)
    return tuple(
        row for row in dataset["episodes"]
        if row.get("split") == split
        and (not hard_only or bool(row.get("hard_reason_codes")))
    )


def _load_dataset(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"dataset schema must be {DATASET_SCHEMA}")
    if not isinstance(value.get("episodes"), list):
        raise ValueError("dataset episodes must be a list")
    return value


def _write_immutable_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite versioned artifact: {target}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--events", action="append")
    build.add_argument(
        "--workspace",
        help="Discover standard ledgers under this root; defaults to cwd when --events is omitted.",
    )
    build.add_argument("--out", required=True)
    build.add_argument("--split-salt", required=True)
    build.add_argument("--hidden-fraction", type=float, default=0.2)
    hard = subparsers.add_parser("hard-cases")
    hard.add_argument("--dataset", required=True)
    hard.add_argument("--out", required=True)
    hard.add_argument("--split", choices=("dev", "hidden"), default="dev")
    replay = subparsers.add_parser("replay")
    replay.add_argument("--dataset", required=True)
    replay.add_argument("--split", choices=("dev", "hidden"), required=True)
    replay.add_argument("--hard-only", action="store_true")
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--workspace", default=".")
    inspect.add_argument(
        "--events", action="append", default=[],
        help="Additional event path, relative to --workspace unless absolute.",
    )
    args = parser.parse_args()
    if args.command == "build":
        if args.workspace is not None or not args.events:
            event_paths = discover_event_paths(
                args.workspace or ".", extra_paths=args.events or (),
            )
        else:
            event_paths = tuple(Path(path) for path in args.events)
        if not event_paths:
            parser.error("no episode event files found; pass --events or --workspace")
        result = build_dataset(
            event_paths, args.out, split_salt=args.split_salt,
            hidden_fraction=args.hidden_fraction,
        )
        print(json.dumps({key: result[key] for key in (
            "schema", "episode_count", "event_count", "hard_episode_count", "split_counts"
        )}, indent=2, sort_keys=True))
    elif args.command == "hard-cases":
        result = build_hard_cases(args.dataset, args.out, split=args.split)
        print(json.dumps({"schema": result["schema"], "episode_count": result["episode_count"]}, indent=2))
    elif args.command == "replay":
        for row in replay_rows(args.dataset, split=args.split, hard_only=args.hard_only):
            print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(
            inspect_workspace(args.workspace, extra_paths=args.events),
            indent=2, sort_keys=True,
        ))


if __name__ == "__main__":
    main()
