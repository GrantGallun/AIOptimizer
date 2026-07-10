#!/usr/bin/env python3
"""Common memory-backend contract and optional third-party adapters.

The contract keeps authorization semantics outside model prompts: a query names
one worker scope, and adapters may return only that scope plus project/global
records. Optional dependencies are imported lazily so the native backend and
tests remain dependency-free.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Awaitable, Protocol

from experiments.brain_runtime.runtime import tokenize
from experiments.brain_runtime.session_runtime import PersistentBrainRuntime


class BackendUnavailable(RuntimeError):
    """An optional backend dependency or service configuration is unavailable."""


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    topic: str
    content: str
    scope: str
    source: str
    kind: str = "semantic"
    key: str | None = None
    value: str | None = None
    confidence: float = 0.5
    utility: float = 0.5
    observed_at: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        fields = cls.__dataclass_fields__
        return cls(**{key: payload[key] for key in fields if key in payload})


@dataclass(frozen=True)
class MemoryHit:
    record: MemoryRecord
    score: float
    backend_metadata: dict[str, Any]


class MemoryBackend(Protocol):
    name: str

    def add(self, record: MemoryRecord) -> None: ...

    def begin_session(self, session_id: str) -> None: ...

    def search(self, query: str, *, scope: str, limit: int) -> list[MemoryHit]: ...

    def close(self) -> None: ...


def visible_scopes(scope: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((scope, "project", "global")))


def lexical_score(query: str, record: MemoryRecord) -> float:
    query_tokens = tokenize(query)
    record_tokens = tokenize(f"{record.topic} {record.content} {record.key or ''} {record.value or ''}")
    if not query_tokens or not record_tokens:
        return 0.0
    overlap = len(query_tokens & record_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(record_tokens))


class NativeMemoryBackend:
    name = "aioptimizer_native"

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.runtime = PersistentBrainRuntime(decay_half_life=6.0, working_memory_limit=256)
        self._records_by_memory_id: dict[str, MemoryRecord] = {}
        self._sessions = 0

    def add(self, record: MemoryRecord) -> None:
        memory = self.runtime.publish_cache(
            record.topic,
            record.content,
            scope=record.scope,
            source=record.source,
            confidence=record.confidence,
            utility=record.utility,
            key=record.key,
            value=record.value,
        )
        self._records_by_memory_id[memory.id] = record

    def begin_session(self, session_id: str) -> None:
        if self._sessions:
            self.runtime = PersistentBrainRuntime.from_snapshot(self.runtime.snapshot())
        self._sessions += 1

    def search(self, query: str, *, scope: str, limit: int) -> list[MemoryHit]:
        memories = self.runtime.read_cache(query, scope=scope, limit=limit)
        return [
            MemoryHit(
                self._records_by_memory_id[memory.id],
                self.runtime.activation(memory, query),
                {"memory_id": memory.id, "uses": memory.uses},
            )
            for memory in memories
        ]

    def close(self) -> None:
        return None


class LexicalMemoryBackend:
    """Dependency-free append store with deterministic cosine token ranking."""

    name = "lexical_append_store"

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.records: list[MemoryRecord] = []

    def add(self, record: MemoryRecord) -> None:
        self.records.append(record)

    def begin_session(self, session_id: str) -> None:
        return None

    def search(self, query: str, *, scope: str, limit: int) -> list[MemoryHit]:
        allowed = set(visible_scopes(scope))
        ranked = sorted(
            enumerate(self.records),
            key=lambda pair: (lexical_score(query, pair[1]), pair[0]),
            reverse=True,
        )
        return [
            MemoryHit(record, lexical_score(query, record), {"insertion_index": index})
            for index, record in ranked
            if record.scope in allowed
        ][:limit]

    def close(self) -> None:
        return None


def deterministic_embeddings(texts: list[str], *, dimensions: int = 128) -> list[list[float]]:
    """Stable signed hashing vectors for a network-free LangGraph baseline."""
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        vectors.append([value / norm for value in vector])
    return vectors


class LangGraphMemoryBackend:
    name = "langgraph_inmemory_hash"

    def __init__(self, project_id: str, *, store: Any | None = None) -> None:
        self.project_id = project_id
        if store is None:
            try:
                from langgraph.store.memory import InMemoryStore
            except ImportError as exc:
                raise BackendUnavailable("LangGraph is not installed; install 'langgraph'") from exc
            store = InMemoryStore(
                index={
                    "embed": deterministic_embeddings,
                    "dims": 128,
                    "fields": ["topic", "content", "key", "value"],
                }
            )
        self.store = store

    def _namespace(self, scope: str) -> tuple[str, ...]:
        return (self.project_id, scope, "memories")

    def add(self, record: MemoryRecord) -> None:
        self.store.put(self._namespace(record.scope), record.id, asdict(record))

    def begin_session(self, session_id: str) -> None:
        return None

    def search(self, query: str, *, scope: str, limit: int) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for visible_scope in visible_scopes(scope):
            for item in self.store.search(self._namespace(visible_scope), query=query, limit=limit):
                record = MemoryRecord.from_dict(dict(item.value))
                hits.append(
                    MemoryHit(
                        record,
                        float(getattr(item, "score", 0.0) or 0.0),
                        {"namespace": list(item.namespace), "key": item.key},
                    )
                )
        return sorted(hits, key=lambda hit: (hit.score, hit.record.observed_at), reverse=True)[:limit]

    def close(self) -> None:
        return None


class Mem0MemoryBackend:
    name = "mem0_oss"

    def __init__(self, project_id: str, *, memory: Any) -> None:
        self.project_id = project_id
        self.memory = memory

    def _entity(self, scope: str) -> str:
        return f"{self.project_id}:{scope}"

    def add(self, record: MemoryRecord) -> None:
        self.memory.add(
            record.content,
            user_id=self._entity(record.scope),
            metadata={"aioptimizer_record": asdict(record)},
            infer=False,
        )

    def begin_session(self, session_id: str) -> None:
        return None

    def search(self, query: str, *, scope: str, limit: int) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for visible_scope in visible_scopes(scope):
            response = self.memory.search(
                query,
                filters={"user_id": self._entity(visible_scope)},
                limit=limit,
            )
            rows = response.get("results", response) if isinstance(response, dict) else response
            for row in rows or []:
                metadata_payload = row.get("metadata", {}).get("aioptimizer_record")
                if metadata_payload:
                    record = MemoryRecord.from_dict(metadata_payload)
                else:
                    record = MemoryRecord(
                        id=str(row.get("id", "unknown")),
                        topic="mem0-result",
                        content=str(row.get("memory", row.get("text", ""))),
                        scope=visible_scope,
                        source="mem0",
                    )
                hits.append(MemoryHit(record, float(row.get("score", 0.0) or 0.0), {"mem0_id": row.get("id")}))
        return sorted(hits, key=lambda hit: (hit.score, hit.record.observed_at), reverse=True)[:limit]

    def close(self) -> None:
        close = getattr(self.memory, "close", None)
        if close:
            close()


class GraphitiMemoryBackend:
    name = "graphiti"

    def __init__(self, project_id: str, *, graph: Any, episode_type_json: Any) -> None:
        self.project_id = project_id
        self.graph = graph
        self.episode_type_json = episode_type_json

    def _group(self, scope: str) -> str:
        return f"{self.project_id}:{scope}"

    def add(self, record: MemoryRecord) -> None:
        self._run(
            self.graph.add_episode(
                name=record.id,
                episode_body=json.dumps(asdict(record), sort_keys=True),
                source=self.episode_type_json,
                source_description="AIOptimizer backend comparison record",
                reference_time=datetime.fromtimestamp(record.observed_at or 0, tz=timezone.utc),
                group_id=self._group(record.scope),
            )
        )

    def begin_session(self, session_id: str) -> None:
        return None

    def search(self, query: str, *, scope: str, limit: int) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for visible_scope in visible_scopes(scope):
            rows = self._run(self.graph.search(query=query, group_id=self._group(visible_scope)))
            for row in (rows or [])[:limit]:
                hits.append(
                    MemoryHit(
                        MemoryRecord(
                            id=str(getattr(row, "uuid", "unknown")),
                            topic="graphiti-fact",
                            content=str(getattr(row, "fact", "")),
                            scope=visible_scope,
                            source="graphiti",
                        ),
                        float(getattr(row, "score", 0.0) or 0.0),
                        {
                            "valid_at": _iso(getattr(row, "valid_at", None)),
                            "invalid_at": _iso(getattr(row, "invalid_at", None)),
                        },
                    )
                )
        return hits[:limit]

    def close(self) -> None:
        close = getattr(self.graph, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                self._run(result)

    @staticmethod
    def _run(awaitable: Awaitable[Any]) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise RuntimeError("GraphitiMemoryBackend sync adapter cannot run inside an active event loop")


def create_backend(
    name: str,
    project_id: str,
    *,
    mem0_config: str | Path | None = None,
) -> MemoryBackend:
    if name == "native":
        return NativeMemoryBackend(project_id)
    if name == "lexical":
        return LexicalMemoryBackend(project_id)
    if name == "langgraph":
        return LangGraphMemoryBackend(project_id)
    if name == "mem0":
        if mem0_config is None:
            raise BackendUnavailable("Mem0 requires --mem0-config to avoid implicit hosted defaults")
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise BackendUnavailable("Mem0 is not installed; install 'mem0ai'") from exc
        config = json.loads(Path(mem0_config).read_text(encoding="utf-8"))
        return Mem0MemoryBackend(project_id, memory=Memory.from_config(config))
    if name == "graphiti":
        try:
            from graphiti_core import Graphiti
            from graphiti_core.nodes import EpisodeType
        except ImportError as exc:
            raise BackendUnavailable("Graphiti is not installed; install 'graphiti-core'") from exc
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        if not all((uri, user, password)):
            raise BackendUnavailable("Graphiti requires NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD")
        graph = Graphiti(uri, user, password)
        return GraphitiMemoryBackend(project_id, graph=graph, episode_type_json=EpisodeType.json)
    raise ValueError(f"unknown backend: {name}")


def backend_version(name: str) -> str | None:
    packages = {"langgraph": "langgraph", "mem0": "mem0ai", "graphiti": "graphiti-core"}
    package = packages.get(name)
    if package is None:
        return None
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None

