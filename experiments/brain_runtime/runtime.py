#!/usr/bin/env python3
"""Deterministic Brain Runtime v0.

This module is intentionally model-free. It tests whether an external cognitive
architecture can provide useful properties before we attach it to any LLM.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


@dataclass
class Evidence:
    source: str
    confidence: float
    observed_at: int
    note: str = ""


@dataclass
class MemoryItem:
    id: str
    topic: str
    content: str
    kind: str = "observation"
    scope: str = "project"
    confidence: float = 0.5
    utility: float = 0.5
    created_at: int = 0
    last_accessed: int = 0
    uses: int = 0
    links: set[str] = field(default_factory=set)
    evidence: list[Evidence] = field(default_factory=list)
    key: str | None = None
    value: str | None = None
    contradicted_by: set[str] = field(default_factory=set)

    @property
    def tokens(self) -> set[str]:
        return tokenize(f"{self.topic} {self.content} {self.key or ''} {self.value or ''}")


@dataclass
class Contradiction:
    topic: str
    key: str
    older_id: str
    newer_id: str
    older_value: str
    newer_value: str
    resolved_to: str


@dataclass
class TaskHook:
    id: str
    prompt: str
    reason: str
    memory_ids: list[str]


class BrainRuntime:
    """Small external memory system with activation, decay, and replay."""

    def __init__(self, decay_half_life: float = 6.0, working_memory_limit: int = 8) -> None:
        self.decay_half_life = decay_half_life
        self.working_memory_limit = working_memory_limit
        self.clock = 0
        self.working_memory: dict[str, MemoryItem] = {}
        self.long_term_memory: dict[str, MemoryItem] = {}
        self.shared_cache: dict[str, MemoryItem] = {}
        self.contradictions: list[Contradiction] = []
        self.task_hooks: list[TaskHook] = []
        self._next_memory_id = 1
        self._next_task_id = 1

    def tick(self, steps: int = 1) -> int:
        self.clock += steps
        return self.clock

    def remember(
        self,
        topic: str,
        content: str,
        *,
        kind: str = "observation",
        scope: str = "project",
        confidence: float = 0.5,
        utility: float = 0.5,
        source: str = "runtime",
        key: str | None = None,
        value: str | None = None,
        links: set[str] | None = None,
        long_term: bool = False,
    ) -> MemoryItem:
        self.tick()
        memory = MemoryItem(
            id=self._new_memory_id(),
            topic=topic,
            content=content,
            kind=kind,
            scope=scope,
            confidence=clamp(confidence),
            utility=clamp(utility),
            created_at=self.clock,
            last_accessed=self.clock,
            links=set(links or set()),
            key=key,
            value=value,
            evidence=[Evidence(source=source, confidence=clamp(confidence), observed_at=self.clock)],
        )
        self._store(memory, long_term=long_term)
        self._detect_contradictions(memory)
        return memory

    def publish_cache(
        self,
        topic: str,
        content: str,
        *,
        scope: str,
        source: str,
        confidence: float = 0.5,
        utility: float = 0.5,
        key: str | None = None,
        value: str | None = None,
    ) -> MemoryItem:
        memory = self.remember(
            topic,
            content,
            kind="cache",
            scope=scope,
            confidence=confidence,
            utility=utility,
            source=source,
            key=key,
            value=value,
        )
        self.shared_cache[f"{scope}:{topic}:{memory.id}"] = memory
        return memory

    def read_cache(self, query: str, *, scope: str, limit: int = 3) -> list[MemoryItem]:
        visible = [item for item in self.shared_cache.values() if item.scope in {scope, "project", "global"}]
        return self._rank(query, visible, limit)

    def retrieve(self, query: str, *, scope: str = "project", limit: int = 5) -> list[MemoryItem]:
        visible = [
            item
            for item in [*self.working_memory.values(), *self.long_term_memory.values(), *self.shared_cache.values()]
            if item.scope in {scope, "project", "global"}
        ]
        return self._rank(query, visible, limit)

    def evidence_score(self, memory: MemoryItem) -> float:
        if not memory.evidence:
            return 0.0
        weighted = sum(item.confidence * self._recency(item.observed_at) for item in memory.evidence)
        return clamp(weighted / len(memory.evidence) + 0.2 * memory.utility)

    def spawn_task(self, prompt: str, *, reason: str, memory_ids: list[str]) -> TaskHook:
        self.tick()
        task = TaskHook(id=f"task-{self._next_task_id:03d}", prompt=prompt, reason=reason, memory_ids=memory_ids)
        self._next_task_id += 1
        self.task_hooks.append(task)
        return task

    def propose_tasks(self, query: str, *, scope: str = "project") -> list[TaskHook]:
        retrieved = self.retrieve(query, scope=scope, limit=3)
        tasks: list[TaskHook] = []
        if any(item.kind == "hypothesis" and item.confidence < 0.7 for item in retrieved):
            ids = [item.id for item in retrieved if item.kind == "hypothesis"]
            tasks.append(self.spawn_task(f"Test uncertain hypothesis for: {query}", reason="low-confidence hypothesis", memory_ids=ids))
        if self.contradictions:
            latest = self.contradictions[-1]
            tasks.append(
                self.spawn_task(
                    f"Resolve contradiction on {latest.topic}:{latest.key}",
                    reason="contradiction pressure",
                    memory_ids=[latest.older_id, latest.newer_id],
                )
            )
        return tasks

    def consolidate(self) -> dict[str, int]:
        promoted = 0
        forgotten = 0
        for memory in list(self.working_memory.values()):
            activation = self.activation(memory, memory.content)
            ambient_activation = self._ambient_activation(memory)
            if memory.uses >= 2 or activation >= 0.75 or self.evidence_score(memory) >= 0.7:
                self.long_term_memory[memory.id] = memory
                self.working_memory.pop(memory.id, None)
                promoted += 1
            elif ambient_activation < 0.2 and memory.utility < 0.35:
                self.working_memory.pop(memory.id, None)
                forgotten += 1
        self._trim_working_memory()
        return {"promoted": promoted, "forgotten": forgotten}

    def activation(self, memory: MemoryItem, query: str) -> float:
        query_tokens = tokenize(query)
        relevance = jaccard(query_tokens, memory.tokens)
        use_bonus = min(memory.uses * 0.04, 0.12)
        link_bonus = self._link_bonus(memory, query_tokens)
        contradiction_penalty = 0.25 if memory.contradicted_by else 0.0
        score = (
            0.60 * relevance
            + 0.08 * self._recency(memory.last_accessed)
            + 0.14 * memory.utility
            + 0.14 * memory.confidence
            + use_bonus
            + link_bonus
            - contradiction_penalty
        )
        return clamp(score)

    def _rank(self, query: str, memories: list[MemoryItem], limit: int) -> list[MemoryItem]:
        ranked = sorted(memories, key=lambda item: (self.activation(item, query), item.created_at), reverse=True)
        selected = ranked[:limit]
        for item in selected:
            item.uses += 1
            item.last_accessed = self.clock
        return selected

    def _store(self, memory: MemoryItem, *, long_term: bool) -> None:
        if long_term:
            self.long_term_memory[memory.id] = memory
        else:
            self.working_memory[memory.id] = memory
            self._trim_working_memory()

    def _trim_working_memory(self) -> None:
        while len(self.working_memory) > self.working_memory_limit:
            weakest = min(self.working_memory.values(), key=lambda item: (self.activation(item, ""), item.utility, item.created_at))
            self.working_memory.pop(weakest.id, None)

    def _detect_contradictions(self, memory: MemoryItem) -> None:
        if not memory.key or memory.value is None:
            return
        candidates = [
            item
            for item in [*self.working_memory.values(), *self.long_term_memory.values(), *self.shared_cache.values()]
            if item.id != memory.id and item.topic == memory.topic and item.key == memory.key and item.value is not None
        ]
        for other in candidates:
            if other.value == memory.value:
                continue
            resolved = self._resolve(memory, other)
            loser = other if resolved.id == memory.id else memory
            loser.contradicted_by.add(resolved.id)
            self.contradictions.append(
                Contradiction(
                    topic=memory.topic,
                    key=memory.key,
                    older_id=other.id if other.created_at < memory.created_at else memory.id,
                    newer_id=memory.id if other.created_at < memory.created_at else other.id,
                    older_value=other.value if other.created_at < memory.created_at else memory.value,
                    newer_value=memory.value if other.created_at < memory.created_at else other.value,
                    resolved_to=resolved.id,
                )
            )

    def _resolve(self, left: MemoryItem, right: MemoryItem) -> MemoryItem:
        left_score = self.evidence_score(left) + 0.05 * left.created_at
        right_score = self.evidence_score(right) + 0.05 * right.created_at
        return left if left_score >= right_score else right

    def _link_bonus(self, memory: MemoryItem, query_tokens: set[str]) -> float:
        linked = [self.long_term_memory.get(link_id) or self.working_memory.get(link_id) for link_id in memory.links]
        if not linked:
            return 0.0
        overlap = max((jaccard(query_tokens, item.tokens) for item in linked if item), default=0.0)
        return min(overlap * 0.12, 0.12)

    def _recency(self, timestamp: int) -> float:
        age = max(self.clock - timestamp, 0)
        return math.exp(-math.log(2) * age / self.decay_half_life)

    def _ambient_activation(self, memory: MemoryItem) -> float:
        use_bonus = min(memory.uses * 0.04, 0.12)
        contradiction_penalty = 0.25 if memory.contradicted_by else 0.0
        return clamp(
            0.24 * self._recency(memory.last_accessed)
            + 0.24 * memory.utility
            + 0.24 * memory.confidence
            + use_bonus
            - contradiction_penalty
        )

    def _new_memory_id(self) -> str:
        memory_id = f"mem-{self._next_memory_id:04d}"
        self._next_memory_id += 1
        return memory_id


class VanillaLoop:
    """Baseline with append-only notes and no decay, contradiction, or replay."""

    def __init__(self) -> None:
        self.notes: list[dict[str, Any]] = []

    def remember(self, topic: str, content: str, **metadata: Any) -> dict[str, Any]:
        note = {"topic": topic, "content": content, **metadata}
        self.notes.append(note)
        return note

    def retrieve(self, query: str, *, scope: str = "project", limit: int = 1) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        matches = [
            note
            for note in self.notes
            if note.get("scope", "project") in {scope, "project", "global"}
            and jaccard(query_tokens, tokenize(f"{note.get('topic', '')} {note.get('content', '')} {note.get('key', '')}")) > 0
        ]
        return matches[:limit]


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
