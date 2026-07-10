#!/usr/bin/env python3
"""Atomic file leases for concurrent agent workspace edits.

The cache protects shared key/value state, while this module protects source paths.
Claims are acquired as one transaction so a worker either owns every requested path
or none of them.  Expired leases are reclaimed during each mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from agent_bus.cache import _cross_process_lock


class WorkspaceConflict(Exception):
    """A path is owned by another worker or a release has stale ownership."""


class WorkspaceClaims:
    def __init__(self, root: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root).resolve()
        self.store = self.root / "workspace_claims.json"
        self.lockfile = self.root / ".workspace_claims.lock"
        self.clock = clock

    def _load(self) -> dict[str, Any]:
        if self.store.exists():
            return json.loads(self.store.read_text(encoding="utf-8"))
        return {"revision": 0, "claims": {}}

    def _save(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.store.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.store)

    def _normalize(self, value: str | Path) -> str:
        candidate = Path(value)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceConflict(f"path escapes workspace: {value}") from exc
        normalized = PurePosixPath(relative.as_posix()).as_posix()
        if normalized in {"", "."}:
            raise WorkspaceConflict("workspace root cannot be claimed")
        return normalized

    def _reclaim_expired(self, state: dict[str, Any], now: float) -> list[str]:
        expired = sorted(path for path, row in state["claims"].items() if row["expires_at"] <= now)
        for path in expired:
            del state["claims"][path]
        return expired

    def claim(self, paths: list[str | Path], *, owner: str, ttl_seconds: float = 900.0) -> dict[str, Any]:
        if not owner.strip():
            raise ValueError("owner must be non-empty")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        normalized = sorted(set(self._normalize(path) for path in paths))
        if not normalized:
            raise ValueError("at least one path is required")
        with _cross_process_lock(self.lockfile):
            state = self._load()
            now = self.clock()
            self._reclaim_expired(state, now)
            conflicts = {
                path: state["claims"][path]["owner"]
                for path in normalized
                if path in state["claims"] and state["claims"][path]["owner"] != owner
            }
            if conflicts:
                detail = ", ".join(f"{path} ({claim_owner})" for path, claim_owner in conflicts.items())
                raise WorkspaceConflict(f"paths already claimed: {detail}")
            state["revision"] += 1
            for path in normalized:
                state["claims"][path] = {
                    "owner": owner,
                    "claimed_at": now,
                    "expires_at": now + ttl_seconds,
                    "revision": state["revision"],
                }
            self._save(state)
            return {"revision": state["revision"], "owner": owner, "paths": normalized}

    def release(self, paths: list[str | Path], *, owner: str) -> list[str]:
        normalized = sorted(set(self._normalize(path) for path in paths))
        with _cross_process_lock(self.lockfile):
            state = self._load()
            self._reclaim_expired(state, self.clock())
            stale = [path for path in normalized if path in state["claims"] and state["claims"][path]["owner"] != owner]
            if stale:
                raise WorkspaceConflict(f"cannot release paths owned by another worker: {', '.join(stale)}")
            released = [path for path in normalized if path in state["claims"]]
            for path in released:
                del state["claims"][path]
            if released:
                state["revision"] += 1
                self._save(state)
            return released

    def list(self) -> dict[str, Any]:
        with _cross_process_lock(self.lockfile):
            state = self._load()
            if self._reclaim_expired(state, self.clock()):
                state["revision"] += 1
                self._save(state)
            return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    sub = parser.add_subparsers(dest="command", required=True)
    claim = sub.add_parser("claim")
    claim.add_argument("paths", nargs="+")
    claim.add_argument("--owner", required=True)
    claim.add_argument("--ttl", type=float, default=900.0)
    release = sub.add_parser("release")
    release.add_argument("paths", nargs="+")
    release.add_argument("--owner", required=True)
    sub.add_parser("list")
    args = parser.parse_args()
    claims = WorkspaceClaims(Path(args.root))
    try:
        if args.command == "claim":
            result = claims.claim(args.paths, owner=args.owner, ttl_seconds=args.ttl)
        elif args.command == "release":
            result = claims.release(args.paths, owner=args.owner)
        else:
            result = claims.list()
        print(json.dumps(result, indent=2, sort_keys=True))
    except WorkspaceConflict as exc:
        print(f"COHERENCE CONFLICT: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
