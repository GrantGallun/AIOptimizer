#!/usr/bin/env python3
"""Versioned native retrieval policy derived from the frozen v1 cache-pollution result.

V1 combined relevance, quality, recency, and global use frequency in one weighted
sum. Repeatedly selected memories could therefore outrank a more relevant record.
V2 makes the control flow explicit: authorization first, then relevance,
contradiction validity, evidence, recency, and finally frequency as a tie-breaker.
"""

from __future__ import annotations

from typing import Any

from experiments.brain_runtime.memory_backends import MemoryHit, NativeMemoryBackend
from experiments.brain_runtime.runtime import MemoryItem, jaccard, tokenize


class PriorityNativeMemoryBackend(NativeMemoryBackend):
    name = "aioptimizer_priority_v2"

    def search(self, query: str, *, scope: str, limit: int) -> list[MemoryHit]:
        visible = [
            memory
            for memory in self.runtime.shared_cache.values()
            if memory.scope in {scope, "project", "global"}
        ]
        ranked = sorted(visible, key=lambda memory: self._rank_key(memory, query), reverse=True)
        selected = ranked[:limit]
        hits: list[MemoryHit] = []
        for memory in selected:
            rank = self._rank_components(memory, query)
            memory.uses += 1
            memory.last_accessed = self.runtime.clock
            hits.append(
                MemoryHit(
                    self._records_by_memory_id[memory.id],
                    rank["relevance"],
                    {
                        "memory_id": memory.id,
                        "rank": rank,
                        "priority_order": [
                            "relevance",
                            "not_contradicted",
                            "evidence",
                            "recency",
                            "uses",
                            "created_at",
                        ],
                    },
                )
            )
        return hits

    def _rank_key(self, memory: MemoryItem, query: str) -> tuple[float, int, float, float, int, int]:
        rank = self._rank_components(memory, query)
        return (
            rank["relevance"],
            rank["not_contradicted"],
            rank["evidence"],
            rank["recency"],
            rank["uses"],
            rank["created_at"],
        )

    def _rank_components(self, memory: MemoryItem, query: str) -> dict[str, Any]:
        return {
            "relevance": jaccard(tokenize(query), memory.tokens),
            "not_contradicted": int(not memory.contradicted_by),
            "evidence": self.runtime.evidence_score(memory),
            "recency": self.runtime._recency(memory.last_accessed),
            "uses": memory.uses,
            "created_at": memory.created_at,
        }

