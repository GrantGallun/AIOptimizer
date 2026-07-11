"""Source-grounded conversation compilation for model context.

Raw turns remain preserved.  A local rewriter may convert them into typed records,
but every record must cite existing turn IDs.  The materializer then builds a
query-specific working set plus a complete lightweight index of the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from agent_bus.context import ContextCompactor, EmbedFn

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
