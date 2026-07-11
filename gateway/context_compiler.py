"""Source-grounded conversation compilation for model context.

Raw turns remain preserved.  A local rewriter may convert them into typed records,
but every record must cite existing turn IDs.  The materializer then builds a
query-specific working set plus a complete lightweight index of the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent_bus.context import ContextCompactor, EmbedFn, _as_vector, _cosine

Record = dict[str, Any]
RewriteFn = Callable[[list[dict[str, str]]], Iterable[Mapping[str, Any]]]


@dataclass(frozen=True)
class CompiledConversation:
    """Preserved transcript and its source-grounded intermediate representation."""

    turns: tuple[dict[str, str], ...]
    records: tuple[Record, ...]


class ConversationCompiler:
    """Compile raw turns, then materialize an indexed context working set."""

    def __init__(self, *, rewrite_fn: RewriteFn | None = None, embed_fn: EmbedFn | None = None):
        self._rewrite_fn = rewrite_fn
        self._selector = ContextCompactor(embed_fn)

    @staticmethod
    def _turns(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        turns = []
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
                raise ValueError("each message needs a supported role and string content")
            turns.append({"id": f"T{index + 1:04d}", "role": role, "content": content})
        return turns

    @staticmethod
    def _lossless_records(turns: list[dict[str, str]]) -> list[Record]:
        return [
            {
                "id": f"R{index + 1:04d}",
                "kind": "verbatim_turn",
                "text": turn["content"],
                "source_ids": [turn["id"]],
                "tags": [turn["role"]],
            }
            for index, turn in enumerate(turns)
        ]

    @staticmethod
    def _validate_records(records: Iterable[Mapping[str, Any]], source_ids: set[str]) -> list[Record]:
        normalized = []
        for index, record in enumerate(records):
            text = record.get("text")
            cited = record.get("source_ids")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("compiled records need non-empty text")
            if not isinstance(cited, list) or not cited or any(item not in source_ids for item in cited):
                raise ValueError("compiled records must cite existing source turn IDs")
            normalized.append(
                {
                    "id": str(record.get("id") or f"R{index + 1:04d}"),
                    "kind": str(record.get("kind") or "note"),
                    "text": text,
                    "source_ids": list(dict.fromkeys(cited)),
                    "tags": [str(tag) for tag in record.get("tags", [])],
                }
            )
        return normalized

    def compile(self, messages: Sequence[Mapping[str, Any]]) -> CompiledConversation:
        turns = self._turns(messages)
        candidate = self._rewrite_fn(turns) if self._rewrite_fn else self._lossless_records(turns)
        records = self._validate_records(candidate, {turn["id"] for turn in turns})
        return CompiledConversation(tuple(turns), tuple(records))

    def materialize(
        self, compiled: CompiledConversation, *, query: str, max_records: int = 8
    ) -> dict[str, Any]:
        """Build a relevant working set while retaining a complete lookup index."""
        if max_records < 0:
            raise ValueError("max_records must be non-negative")
        chunks = [
            {"id": record["id"], "text": record["text"], "record": record}
            for record in compiled.records
        ]
        selected = self._selector.select(query, chunks, min(max_records, len(chunks)))
        working_set = [chunk["record"] for chunk in selected]
        return {
            "schema": "aioptimizer.compiled-context.v1",
            "query": query,
            "working_set": working_set,
            "index": [
                {
                    "id": record["id"],
                    "kind": record["kind"],
                    "source_ids": record["source_ids"],
                    "tags": record["tags"],
                }
                for record in compiled.records
            ],
        }

    def organize(
        self,
        compiled: CompiledConversation,
        *,
        query: str,
        similarity_threshold: float = 0.72,
        max_records: int = 8,
    ) -> dict[str, Any]:
        """Attention-group records, rank groups, and pin load-bearing turns.

        Clustering is deterministic connected-components over pairwise cosine
        similarity.  System turns and the latest user turn are always present in
        the working set; attention ranking cannot displace instructions or the
        active request.
        """
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if max_records < 0:
            raise ValueError("max_records must be non-negative")
        records = list(compiled.records)
        if not records:
            return {
                "schema": "aioptimizer.attention-context.v1",
                "query": query,
                "pinned": [],
                "clusters": [],
                "index": [],
            }

        vectors = [_as_vector(value) for value in self._selector._embed(
            [query] + [record["text"] for record in records]
        )]
        if len(vectors) != len(records) + 1:
            raise ValueError("embed_fn must return one vector per input text")
        query_vector, record_vectors = vectors[0], vectors[1:]

        parent = list(range(len(records)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for left in range(len(records)):
            for right in range(left + 1, len(records)):
                if _cosine(record_vectors[left], record_vectors[right]) >= similarity_threshold:
                    union(left, right)

        grouped: dict[int, list[int]] = {}
        for index in range(len(records)):
            grouped.setdefault(find(index), []).append(index)
        ranked_groups = sorted(
            grouped.values(),
            key=lambda group: (-max(_cosine(query_vector, record_vectors[i]) for i in group), group[0]),
        )

        role_by_source = {turn["id"]: turn["role"] for turn in compiled.turns}
        latest_user = next(
            (turn["id"] for turn in reversed(compiled.turns) if turn["role"] == "user"), None
        )
        pinned_ids = {
            record["id"]
            for record in records
            if latest_user in record["source_ids"]
            or any(role_by_source[source] == "system" for source in record["source_ids"])
        }
        pinned = [record for record in records if record["id"] in pinned_ids]
        remaining = max(0, max_records - len(pinned))
        clusters = []
        for group in ranked_groups:
            members = [records[index] for index in group if records[index]["id"] not in pinned_ids]
            members = members[:remaining]
            if members:
                clusters.append(
                    {
                        "score": round(max(_cosine(query_vector, record_vectors[i]) for i in group), 4),
                        "records": members,
                    }
                )
                remaining -= len(members)
            if remaining == 0:
                break
        return {
            "schema": "aioptimizer.attention-context.v1",
            "query": query,
            "pinned": pinned,
            "clusters": clusters,
            "index": [
                {"id": record["id"], "kind": record["kind"], "source_ids": record["source_ids"]}
                for record in records
            ],
        }

    @staticmethod
    def _render_record(record: Mapping[str, Any]) -> str:
        sources = ",".join(record["source_ids"])
        return f"[{record['id']}|{record['kind']}|source:{sources}]\n{record['text']}"

    @classmethod
    def _fit_sections(
        cls,
        pinned: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        budget_chars: int,
    ) -> str:
        if budget_chars < 0:
            raise ValueError("budget_chars must be non-negative")
        pinned_blocks = [cls._render_record(record) for record in pinned]
        required = "\n\n".join(pinned_blocks)
        if len(required) > budget_chars:
            raise ValueError("budget is too small for pinned instructions and active request")
        blocks = list(pinned_blocks)
        used = len(required)
        for record in candidates:
            block = cls._render_record(record)
            separator = 2 if blocks else 0
            if used + separator + len(block) <= budget_chars:
                blocks.append(block)
                used += separator + len(block)
        return "\n\n".join(blocks)

    def render_raw(self, compiled: CompiledConversation, *, budget_chars: int) -> str:
        """Render chronological history under the same pinning and budget rules."""
        role_by_source = {turn["id"]: turn["role"] for turn in compiled.turns}
        latest_user = next(
            (turn["id"] for turn in reversed(compiled.turns) if turn["role"] == "user"), None
        )
        pinned = [
            record
            for record in compiled.records
            if latest_user in record["source_ids"]
            or any(role_by_source[source] == "system" for source in record["source_ids"])
        ]
        pinned_ids = {record["id"] for record in pinned}
        candidates = [record for record in compiled.records if record["id"] not in pinned_ids]
        return self._fit_sections(pinned, candidates, budget_chars)

    def render_organized(self, organized: Mapping[str, Any], *, budget_chars: int) -> str:
        """Render attention-ranked clusters under an exact character ceiling."""
        candidates = [
            record
            for cluster in organized.get("clusters", [])
            for record in cluster.get("records", [])
        ]
        return self._fit_sections(organized.get("pinned", []), candidates, budget_chars)
