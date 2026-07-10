#!/usr/bin/env python3
"""shared cache — a coherent shared-memory layer for Fable (Claude) and Codex.

The mental model is two CPU cores with a shared cache. There is no live shared RAM
between two turn-based agents, so — exactly like real cores — each keeps a *local* view
and a small coherence protocol keeps them consistent:

  * one authoritative store (`cache.json`) with a monotonic global `rev` (revision);
  * every key holds a value + version + last writer + scope + provenance;
  * **pull** before you act (refresh your cache lines: what changed since your last rev);
  * **write-through** on every change (set is immediate, no write-back delay);
  * **compare-and-set** (`--expect-version`) detects contention (a failed CAS = another
    core touched the line — re-pull and retry);
  * **claim/release** takes exclusive ownership of a line while you modify it (MESI's
    Modified/Exclusive state) so two cores don't clobber a hot key.

`SHARED.md` renders the whole cache as a live dashboard — keep it open to *see* the shared
memory. Dependency-free (stdlib). Both agents use: pull / get / set / claim / release / view.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Allow direct-script invocation (python agent_bus/cache.py ...) as well as module use.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent_bus.bus import Bus, _now

AGENTS = ("fable", "codex", "user")


class CacheConflict(Exception):
    """Raised on a coherence violation: stale compare-and-set or a line owned by another core."""


@contextmanager
def _cross_process_lock(path: Path, *, timeout: float = 10.0):
    """Exclusive lock held for a whole read-modify-write, so two real drivers cannot lose an
    update (the race Codex flagged: both load rev N, both save N+1, last-writer-wins).

    Uses the OS advisory lock (fcntl/msvcrt) on a handle, so it releases automatically if a
    process crashes mid-transaction — no stale lockfiles.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + timeout
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"cache lock timeout after {timeout}s")
                    time.sleep(0.01)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.store = self.root / "cache.json"
        self.shared = self.root / "SHARED.md"
        self.lockfile = self.root / ".cache.lock"

    def _lock(self):
        return _cross_process_lock(self.lockfile)

    # -- storage (atomic write-through) --------------------------------------
    def _load(self) -> dict[str, Any]:
        if self.store.exists():
            return json.loads(self.store.read_text(encoding="utf-8"))
        return {"rev": 0, "entries": {}, "locks": {}}

    def _save(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.store)  # atomic: readers never see a half-written cache
        self.render(state)

    # -- reads ---------------------------------------------------------------
    def get(self, key: str) -> dict[str, Any] | None:
        return self._load()["entries"].get(key)

    def keys(self) -> list[str]:
        return sorted(self._load()["entries"])

    def pull(self, *, since_rev: int = 0) -> dict[str, Any]:
        """Refresh: entries whose rev is newer than the caller's last-seen rev."""
        state = self._load()
        changed = sorted(
            (e for e in state["entries"].values() if e["rev"] > since_rev),
            key=lambda e: e["rev"],
        )
        return {"rev": state["rev"], "changed": changed, "locks": state["locks"]}

    # -- writes (coherence-checked) ------------------------------------------
    def set(
        self,
        key: str,
        value: str,
        *,
        writer: str,
        expect_version: int | None = None,
        scope: str = "shared",
        note: str = "",
    ) -> dict[str, Any]:
        with self._lock():
            state = self._load()
            current = state["entries"].get(key)
            lock = state["locks"].get(key)
            if lock and lock["holder"] != writer:
                raise CacheConflict(f"key '{key}' is claimed by {lock['holder']}; ask it to release.")
            current_version = current["version"] if current else 0
            if expect_version is not None and expect_version != current_version:
                raise CacheConflict(
                    f"stale write on '{key}': you expected v{expect_version}, cache is v{current_version}. Re-pull."
                )
            state["rev"] += 1
            entry = {
                "key": key,
                "value": value,
                "version": current_version + 1,
                "writer": writer,
                "scope": scope,
                "note": note,
                "ts": _now(),
                "rev": state["rev"],
            }
            state["entries"][key] = entry
            self._save(state)
            return entry

    def claim(self, key: str, *, writer: str) -> dict[str, Any]:
        with self._lock():
            state = self._load()
            lock = state["locks"].get(key)
            if lock and lock["holder"] != writer:
                raise CacheConflict(f"key '{key}' already claimed by {lock['holder']}.")
            state["rev"] += 1
            state["locks"][key] = {"holder": writer, "ts": _now(), "rev": state["rev"]}
            self._save(state)
            return state["locks"][key]

    def release(self, key: str, *, writer: str) -> bool:
        with self._lock():
            state = self._load()
            lock = state["locks"].get(key)
            if not lock:
                return False
            if lock["holder"] != writer:
                raise CacheConflict(f"key '{key}' is held by {lock['holder']}, not {writer}.")
            del state["locks"][key]
            state["rev"] += 1
            self._save(state)
            return True

    # -- dashboard -----------------------------------------------------------
    def render(self, state: dict[str, Any] | None = None) -> str:
        state = state if state is not None else self._load()
        lines = [
            "# Shared Cache — Fable ⇄ Codex",
            "",
            f"_Coherent shared memory. `rev {state['rev']}` · {len(state['entries'])} keys · "
            f"updated {_now()}._ Source of truth: `cache.json`.",
            "",
            "| key | value | v | writer | scope | updated |",
            "| --- | --- | --: | --- | --- | --- |",
        ]
        for key in sorted(state["entries"]):
            e = state["entries"][key]
            value = e["value"].replace("|", "\\|").replace("\n", " ")
            if len(value) > 90:
                value = value[:87] + "…"
            lines.append(
                f"| `{key}` | {value} | {e['version']} | {e['writer']} | {e['scope']} | {e['ts']} |"
            )
        if state["locks"]:
            lines += ["", "**Claimed lines (exclusive ownership):**"]
            for key, lock in sorted(state["locks"].items()):
                lines.append(f"- `{key}` → held by **{lock['holder']}** since {lock['ts']}")
        # A glance at the running conversation stream that accompanies the state.
        recent = Bus(self.root).messages()[-5:]
        if recent:
            lines += ["", "**Recent channel messages:**"]
            for m in recent:
                lines.append(f"- `{m['id']}` {m['from']} → {m['to']} ({m['type']}): {m['body'][:80]}")
        text = "\n".join(lines) + "\n"
        self.shared.write_text(text, encoding="utf-8")
        return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pull = sub.add_parser("pull", help="Refresh your view: entries changed since --since rev.")
    p_pull.add_argument("--since", type=int, default=0)
    p_pull.set_defaults(func=lambda c, a: print(json.dumps(c.pull(since_rev=a.since), indent=2, ensure_ascii=False)))

    p_get = sub.add_parser("get", help="Read one key.")
    p_get.add_argument("key")
    p_get.set_defaults(func=lambda c, a: print(json.dumps(c.get(a.key), indent=2, ensure_ascii=False)))

    p_set = sub.add_parser("set", help="Write-through a key.")
    p_set.add_argument("key")
    p_set.add_argument("value")
    p_set.add_argument("--writer", required=True, choices=AGENTS)
    p_set.add_argument("--expect-version", type=int, default=None, help="Compare-and-set: fail if the line moved.")
    p_set.add_argument("--scope", default="shared")
    p_set.add_argument("--note", default="")
    p_set.set_defaults(
        func=lambda c, a: print(
            f"set {a.key} -> v{c.set(a.key, a.value, writer=a.writer, expect_version=a.expect_version, scope=a.scope, note=a.note)['version']}"
            f" (rev {c._load()['rev']}). SHARED.md updated."
        )
    )

    p_claim = sub.add_parser("claim", help="Take exclusive ownership of a line.")
    p_claim.add_argument("key")
    p_claim.add_argument("--writer", required=True, choices=AGENTS)
    p_claim.set_defaults(func=lambda c, a: print(f"claimed {a.key} for {a.writer}: {c.claim(a.key, writer=a.writer)}"))

    p_rel = sub.add_parser("release", help="Release a claimed line.")
    p_rel.add_argument("key")
    p_rel.add_argument("--writer", required=True, choices=AGENTS)
    p_rel.set_defaults(func=lambda c, a: print(f"released {a.key}: {c.release(a.key, writer=a.writer)}"))

    p_keys = sub.add_parser("keys", help="List keys.")
    p_keys.set_defaults(func=lambda c, a: print("\n".join(c.keys()) or "(empty)"))

    p_view = sub.add_parser("view", help="Regenerate SHARED.md.")
    p_view.set_defaults(func=lambda c, a: (c.render(), print(f"rendered {c.shared}")))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    cache = Cache(Path(args.root))
    try:
        args.func(cache, args)
    except CacheConflict as exc:
        print(f"COHERENCE CONFLICT: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
