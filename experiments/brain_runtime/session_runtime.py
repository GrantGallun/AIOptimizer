#!/usr/bin/env python3
"""Versioned cross-session persistence for :mod:`brain_runtime.runtime`.

``runtime.py`` remains the deterministic v0 memory-policy implementation used by
the frozen benchmarks.  This module layers a JSON snapshot boundary over it so a
worker can carry governed memory into a later process or session without changing
v0 retrieval, contradiction, consolidation, or scope semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.brain_runtime.runtime import (
    BrainRuntime,
    Contradiction,
    Evidence,
    MemoryItem,
    TaskHook,
)


SNAPSHOT_SCHEMA = "brain-runtime-session-v1"


class PersistentBrainRuntime(BrainRuntime):
    """A ``BrainRuntime`` whose complete state can cross a process boundary."""

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable, versioned representation of runtime state."""
        memories: dict[str, MemoryItem] = {}
        for store in (self.working_memory, self.long_term_memory, self.shared_cache):
            for memory in store.values():
                existing = memories.get(memory.id)
                if existing is not None and existing is not memory:
                    raise ValueError(f"memory id {memory.id!r} refers to multiple objects")
                memories[memory.id] = memory

        return {
            "schema": SNAPSHOT_SCHEMA,
            "config": {
                "decay_half_life": self.decay_half_life,
                "working_memory_limit": self.working_memory_limit,
            },
            "state": {
                "clock": self.clock,
                "next_memory_id": self._next_memory_id,
                "next_task_id": self._next_task_id,
            },
            "memories": [self._memory_to_dict(memories[key]) for key in sorted(memories)],
            "stores": {
                "working_memory": {key: item.id for key, item in sorted(self.working_memory.items())},
                "long_term_memory": {key: item.id for key, item in sorted(self.long_term_memory.items())},
                "shared_cache": {key: item.id for key, item in sorted(self.shared_cache.items())},
            },
            "contradictions": [
                {
                    "topic": item.topic,
                    "key": item.key,
                    "older_id": item.older_id,
                    "newer_id": item.newer_id,
                    "older_value": item.older_value,
                    "newer_value": item.newer_value,
                    "resolved_to": item.resolved_to,
                }
                for item in self.contradictions
            ],
            "task_hooks": [
                {
                    "id": item.id,
                    "prompt": item.prompt,
                    "reason": item.reason,
                    "memory_ids": list(item.memory_ids),
                }
                for item in self.task_hooks
            ],
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "PersistentBrainRuntime":
        """Restore a snapshot, rejecting unknown schemas and dangling store refs."""
        if payload.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError(f"unsupported snapshot schema: {payload.get('schema')!r}")

        config = payload.get("config")
        state = payload.get("state")
        rows = payload.get("memories")
        stores = payload.get("stores")
        if not isinstance(config, dict) or not isinstance(state, dict):
            raise ValueError("snapshot config/state must be objects")
        if not isinstance(rows, list) or not isinstance(stores, dict):
            raise ValueError("snapshot memories/stores have invalid types")

        runtime = cls(
            decay_half_life=float(config["decay_half_life"]),
            working_memory_limit=int(config["working_memory_limit"]),
        )
        memories: dict[str, MemoryItem] = {}
        for row in rows:
            memory = cls._memory_from_dict(row)
            if memory.id in memories:
                raise ValueError(f"duplicate memory id in snapshot: {memory.id}")
            memories[memory.id] = memory

        def restore_store(name: str) -> dict[str, MemoryItem]:
            index = stores.get(name)
            if not isinstance(index, dict):
                raise ValueError(f"snapshot store {name!r} must be an object")
            restored: dict[str, MemoryItem] = {}
            for key, memory_id in index.items():
                try:
                    restored[str(key)] = memories[str(memory_id)]
                except KeyError as exc:
                    raise ValueError(f"snapshot store {name!r} references missing memory {memory_id!r}") from exc
            return restored

        runtime.working_memory = restore_store("working_memory")
        runtime.long_term_memory = restore_store("long_term_memory")
        runtime.shared_cache = restore_store("shared_cache")
        runtime.contradictions = [Contradiction(**row) for row in payload.get("contradictions", [])]
        runtime.task_hooks = [TaskHook(**row) for row in payload.get("task_hooks", [])]
        runtime.clock = int(state["clock"])
        runtime._next_memory_id = int(state["next_memory_id"])
        runtime._next_task_id = int(state["next_task_id"])
        return runtime

    def save(self, path: str | Path) -> Path:
        """Atomically write a snapshot and return its resolved path."""
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PersistentBrainRuntime":
        """Load a snapshot previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("snapshot root must be an object")
        return cls.from_snapshot(payload)

    @staticmethod
    def _memory_to_dict(memory: MemoryItem) -> dict[str, Any]:
        return {
            "id": memory.id,
            "topic": memory.topic,
            "content": memory.content,
            "kind": memory.kind,
            "scope": memory.scope,
            "confidence": memory.confidence,
            "utility": memory.utility,
            "created_at": memory.created_at,
            "last_accessed": memory.last_accessed,
            "uses": memory.uses,
            "links": sorted(memory.links),
            "evidence": [
                {
                    "source": item.source,
                    "confidence": item.confidence,
                    "observed_at": item.observed_at,
                    "note": item.note,
                }
                for item in memory.evidence
            ],
            "key": memory.key,
            "value": memory.value,
            "contradicted_by": sorted(memory.contradicted_by),
        }

    @staticmethod
    def _memory_from_dict(row: Any) -> MemoryItem:
        if not isinstance(row, dict):
            raise ValueError("snapshot memory entries must be objects")
        return MemoryItem(
            id=str(row["id"]),
            topic=str(row["topic"]),
            content=str(row["content"]),
            kind=str(row["kind"]),
            scope=str(row["scope"]),
            confidence=float(row["confidence"]),
            utility=float(row["utility"]),
            created_at=int(row["created_at"]),
            last_accessed=int(row["last_accessed"]),
            uses=int(row["uses"]),
            links={str(value) for value in row.get("links", [])},
            evidence=[Evidence(**item) for item in row.get("evidence", [])],
            key=row.get("key"),
            value=row.get("value"),
            contradicted_by={str(value) for value in row.get("contradicted_by", [])},
        )

