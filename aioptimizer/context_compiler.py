"""Source-grounded conversation compilation for model context.

Raw turns remain preserved.  A local rewriter may convert them into typed records,
but every record must cite existing turn IDs.  The materializer then builds a
query-specific working set plus a complete lightweight index of the conversation.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from aioptimizer.selection import ContextCompactor

from .encoder import EmbedFn, as_vector, cosine, embed_texts

# Backward-compatible research aliases. Product code uses the public encoder names.
_as_vector = as_vector
_cosine = cosine

Record = dict[str, Any]
RewriteFn = Callable[[list[dict[str, str]]], Iterable[Mapping[str, Any]]]
DEFAULT_DENY_PATTERNS = (
    r"private\s*[\u2014-]?\s*never repeat",
    r"do not (share|repeat|reveal)",
    r"confidential",
)
COMPILER_SCHEMA = "aioptimizer.context-ir.v1"
RECORD_KINDS = frozenset(
    {
        "instruction", "constraint", "decision", "fact", "evidence", "result",
        "open_question", "artifact", "preference", "hypothesis", "note", "verbatim_turn",
    }
)
KIND_ORDER = (
    "instruction", "constraint", "decision", "fact", "evidence", "result",
    "preference", "hypothesis", "open_question", "artifact", "note", "verbatim_turn",
)
AUTHORITIES = frozenset({"source", "derived", "inferred"})


@dataclass(frozen=True)
class CompiledConversation:
    """Preserved transcript and its source-grounded intermediate representation."""

    turns: tuple[dict[str, str], ...]
    records: tuple[Record, ...]


class ConversationCompiler:
    """Compile raw turns, then materialize an indexed context working set."""

    def __init__(
        self,
        *,
        rewrite_fn: RewriteFn | None = None,
        embed_fn: EmbedFn | None = None,
        deny_patterns: Sequence[str] = DEFAULT_DENY_PATTERNS,
        embed_cache_entries: int = 4096,
    ):
        if embed_cache_entries < 0:
            raise ValueError("embed_cache_entries must be non-negative")
        self._rewrite_fn = rewrite_fn
        self._raw_selector = ContextCompactor(embed_fn or embed_texts)
        self._embed_cache_entries = embed_cache_entries
        self._embed_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._embed_cache_lock = threading.Lock()
        self._embed_cache_hits = 0
        self._embed_cache_misses = 0
        self._selector = ContextCompactor(self._cached_embed)
        self._deny_patterns = tuple(re.compile(pattern, re.IGNORECASE) for pattern in deny_patterns)

    def _cached_embed(self, texts: list[str]) -> list[list[float]]:
        """Incrementally embed exact text once; stable IDs/ties remain outside the cache."""
        if self._embed_cache_entries == 0:
            return [_as_vector(value) for value in self._raw_selector._embed(texts)]
        missing = []
        resolved: dict[str, list[float]] = {}
        with self._embed_cache_lock:
            for text in dict.fromkeys(texts):
                if text in self._embed_cache:
                    self._embed_cache_hits += texts.count(text)
                    self._embed_cache.move_to_end(text)
                    resolved[text] = list(self._embed_cache[text])
                else:
                    missing.append(text)
                    self._embed_cache_misses += texts.count(text)
        if missing:
            values = [_as_vector(value) for value in self._raw_selector._embed(missing)]
            if len(values) != len(missing):
                raise ValueError("embed_fn must return one vector per input text")
            with self._embed_cache_lock:
                for text, value in zip(missing, values):
                    resolved[text] = value
                    self._embed_cache[text] = value
                    self._embed_cache.move_to_end(text)
                while len(self._embed_cache) > self._embed_cache_entries:
                    self._embed_cache.popitem(last=False)
        return [list(resolved[text]) for text in texts]

    def embedding_cache_stats(self) -> dict[str, int]:
        with self._embed_cache_lock:
            return {
                "entries": len(self._embed_cache),
                "hits": self._embed_cache_hits,
                "misses": self._embed_cache_misses,
                "capacity": self._embed_cache_entries,
            }

    def relevance_profile(
        self, compiled: CompiledConversation, *, query: str
    ) -> dict[str, float | int]:
        """Deterministic retrieval-confidence profile excluding pinned request/instructions."""
        pinned_ids = {record["id"] for record in self.pinned_records(compiled)}
        records = [
            record for record in self._public_records(compiled) if record["id"] not in pinned_ids
        ]
        if not records:
            return {"candidates": 0, "peak": 0.0, "margin": 0.0, "mean": 0.0}
        vectors = [
            _as_vector(value)
            for value in self._selector._embed([query] + [record["text"] for record in records])
        ]
        if len(vectors) != len(records) + 1:
            raise ValueError("embed_fn must return one vector per input text")
        scores = sorted(
            (_cosine(vectors[0], vector) for vector in vectors[1:]), reverse=True
        )
        return {
            "candidates": len(records),
            "peak": round(scores[0], 6),
            "margin": round(scores[0] - scores[1], 6) if len(scores) > 1 else round(scores[0], 6),
            "mean": round(sum(scores) / len(scores), 6),
        }

    def choose_context_mode(
        self, compiled: CompiledConversation, *, query: str, min_relevance: float = 0.5
    ) -> tuple[str, dict[str, float | int]]:
        """Use attention only when history has a strong semantic address for the request."""
        if not -1.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between -1 and 1")
        profile = self.relevance_profile(compiled, query=query)
        mode = "attention" if float(profile["peak"]) >= min_relevance else "raw"
        return mode, profile

    def _is_denied(self, text: str) -> bool:
        # Treat Markdown-style double hyphens as the single dash allowed by the
        # fixed policy pattern, while preserving the original text everywhere else.
        normalized = re.sub(r"-{2,}", "-", text)
        return any(
            pattern.search(text) is not None or pattern.search(normalized) is not None
            for pattern in self._deny_patterns
        )

    def _public_records(self, compiled: CompiledConversation) -> list[Record]:
        return [record for record in compiled.records if not self._is_denied(record["text"])]

    def _index_entry(self, record: Mapping[str, Any], *, include_tags: bool) -> dict[str, Any]:
        if self._is_denied(str(record["text"])):
            return {
                "id": record["id"],
                "kind": record["kind"],
                "text": "[redacted: privacy]",
            }
        entry = {
            "id": record["id"],
            "kind": record["kind"],
            "source_ids": record["source_ids"],
        }
        if include_tags:
            entry["tags"] = record["tags"]
        return entry

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
        role_kinds = {
            "system": "instruction", "user": "note", "assistant": "result", "tool": "evidence",
        }
        return [
            {
                "id": f"R{index + 1:04d}",
                "kind": role_kinds[turn["role"]],
                "text": turn["content"],
                "source_ids": [turn["id"]],
                "tags": [turn["role"]],
                "authority": "source",
                "binding": turn["role"] in {"system", "user"},
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
            kind = str(record.get("kind") or "note")
            authority = str(record.get("authority") or "inferred")
            binding = bool(record.get("binding", False))
            if kind not in RECORD_KINDS:
                raise ValueError(f"unsupported record kind: {kind}")
            if authority not in AUTHORITIES:
                raise ValueError(f"unsupported record authority: {authority}")
            if binding and authority == "inferred":
                raise ValueError("inferred records cannot be binding")
            normalized.append(
                {
                    "id": str(record.get("id") or f"R{index + 1:04d}"),
                    "kind": kind,
                    "text": text,
                    "source_ids": list(dict.fromkeys(cited)),
                    "tags": [str(tag) for tag in record.get("tags", [])],
                    "authority": authority,
                    "binding": binding,
                }
            )
        ids = [record["id"] for record in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("compiled record IDs must be unique")
        return normalized

    def compile(self, messages: Sequence[Mapping[str, Any]]) -> CompiledConversation:
        turns = self._turns(messages)
        candidate = self._rewrite_fn(turns) if self._rewrite_fn else self._lossless_records(turns)
        records = self._validate_records(candidate, {turn["id"] for turn in turns})
        return CompiledConversation(tuple(turns), tuple(records))

    @staticmethod
    def canonical_bytes(value: Any) -> bytes:
        """Canonical UTF-8 encoding used for replay hashes and exact caches."""
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def canonical_snapshot(self, compiled: CompiledConversation) -> dict[str, Any]:
        """Return the versioned, serialization-safe deterministic context IR."""
        return {
            "schema": COMPILER_SCHEMA,
            "turns": [dict(turn) for turn in compiled.turns],
            "records": [dict(record) for record in compiled.records],
        }

    def fingerprint(self, compiled: CompiledConversation) -> str:
        """Content address for byte-identical replay verification."""
        payload = self.canonical_bytes(self.canonical_snapshot(compiled))
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def audit_integrity(self, compiled: CompiledConversation) -> dict[str, Any]:
        """Verify provenance, extractive source fidelity, and active-request preservation."""
        turns = {turn["id"]: turn for turn in compiled.turns}
        failures: list[dict[str, str]] = []
        for record in compiled.records:
            missing = [source for source in record["source_ids"] if source not in turns]
            if missing:
                failures.append({"record": record["id"], "reason": "missing_source"})
                continue
            if record["authority"] == "source" and not any(
                record["text"] == turns[source]["content"] for source in record["source_ids"]
            ):
                failures.append({"record": record["id"], "reason": "source_text_changed"})
        latest_user = next(
            (turn for turn in reversed(compiled.turns) if turn["role"] == "user"), None
        )
        active_preserved = latest_user is None or any(
            record["authority"] == "source"
            and latest_user["id"] in record["source_ids"]
            and record["text"] == latest_user["content"]
            for record in compiled.records
        )
        if not active_preserved:
            failures.append({"record": "active_request", "reason": "not_preserved_verbatim"})
        return {
            "ok": not failures,
            "active_request_preserved": active_preserved,
            "failures": failures,
            "fingerprint": self.fingerprint(compiled),
        }

    def record_heat(
        self,
        records: Sequence[Mapping[str, Any]],
        queries: Sequence[str],
    ) -> dict[str, float]:
        """Accumulate non-negative record/query similarity using the context encoder."""
        heat = {str(record["id"]): 0.0 for record in records}
        public_records = [
            record for record in records if not self._is_denied(str(record["text"]))
        ]
        if not public_records or not queries:
            return heat

        vectors = [
            _as_vector(value)
            for value in self._selector._embed(
                [record["text"] for record in public_records] + list(queries)
            )
        ]
        expected = len(public_records) + len(queries)
        if len(vectors) != expected:
            raise ValueError("embed_fn must return one vector per input text")
        record_vectors = vectors[: len(public_records)]
        query_vectors = vectors[len(public_records) :]
        for record, record_vector in zip(public_records, record_vectors):
            heat[str(record["id"])] = sum(
                max(0.0, _cosine(record_vector, query_vector))
                for query_vector in query_vectors
            )
        return heat

    def materialize(
        self,
        compiled: CompiledConversation,
        *,
        query: str,
        max_records: int = 8,
        heat: Mapping[str, float] | None = None,
        heat_weight: float = 0.0,
    ) -> dict[str, Any]:
        """Build a relevant working set while retaining a complete lookup index."""
        if max_records < 0:
            raise ValueError("max_records must be non-negative")
        public_records = self._public_records(compiled)
        chunks = [
            {"id": record["id"], "text": record["text"], "record": record}
            for record in public_records
        ]
        limit = min(max_records, len(chunks))
        if heat_weight <= 0.0 or not chunks:
            selected = self._selector.select(query, chunks, limit)
        else:
            vectors = [
                _as_vector(value)
                for value in self._selector._embed([query] + [chunk["text"] for chunk in chunks])
            ]
            if len(vectors) != len(chunks) + 1:
                raise ValueError("embed_fn must return one vector per input text")
            relevance = [_cosine(vectors[0], vector) for vector in vectors[1:]]
            heat_values = [
                max(0.0, float((heat or {}).get(str(chunk["id"]), 0.0)))
                for chunk in chunks
            ]
            max_heat = max(heat_values, default=0.0)
            normalized_heat = [value / max_heat if max_heat else 0.0 for value in heat_values]
            scores = [
                (1.0 - heat_weight) * relevance_score + heat_weight * heat_score
                for relevance_score, heat_score in zip(relevance, normalized_heat)
            ]
            order = sorted(range(len(chunks)), key=lambda index: (-scores[index], index))
            selected = [chunks[index] for index in order[:limit]]
        working_set = [chunk["record"] for chunk in selected]
        return {
            "schema": "aioptimizer.compiled-context.v1",
            "query": query,
            "working_set": working_set,
            "index": [self._index_entry(record, include_tags=True) for record in compiled.records],
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
        records = self._public_records(compiled)
        if not records:
            return {
                "schema": "aioptimizer.attention-context.v1",
                "query": query,
                "pinned": [],
                "clusters": [],
                "index": [
                    self._index_entry(record, include_tags=False)
                    for record in compiled.records
                ],
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
                self._index_entry(record, include_tags=False)
                for record in compiled.records
            ],
        }

    @staticmethod
    def _render_record(record: Mapping[str, Any]) -> str:
        sources = ",".join(record["source_ids"])
        return f"[{record['id']}|{record['kind']}|source:{sources}]\n{record['text']}"

    @classmethod
    def render_record_set(
        cls, records: Sequence[Mapping[str, Any]], *, structured: bool = False
    ) -> str:
        """Render an exact record set flat or in deterministic typed sections."""
        if not structured:
            return "\n\n".join(cls._render_record(record) for record in records)
        grouped = {kind: [] for kind in KIND_ORDER}
        for record in records:
            grouped[str(record["kind"])].append(record)
        sections = []
        for kind in KIND_ORDER:
            if grouped[kind]:
                body = "\n\n".join(cls._render_record(record) for record in grouped[kind])
                sections.append(f"## {kind.upper()}\n{body}")
        return "\n\n".join(sections)

    def render_governed_record_set(
        self,
        compiled: CompiledConversation,
        records: Sequence[Mapping[str, Any]],
    ) -> str:
        """Validate typed governance internally, then emit the minimal flat model view.

        Authority, binding, and provenance remain enforced by the IR but do not
        reorder or decorate the model-facing record stream.  This deliberately
        makes presentation byte-identical to attention-only rendering.
        """
        audit = self.audit_integrity(compiled)
        if not audit["ok"]:
            raise ValueError(f"context integrity failed: {audit['failures']}")
        known = {record["id"] for record in compiled.records}
        if any(str(record["id"]) not in known for record in records):
            raise ValueError("governed render contains a record outside the compiled IR")
        return self.render_record_set(records, structured=False)

    @classmethod
    def matched_record_pair(
        cls,
        pinned: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        *,
        budget_chars: int,
    ) -> tuple[str, str, list[Mapping[str, Any]]]:
        """Fit one information-identical record set to flat and structured renderings."""
        if budget_chars < 0:
            raise ValueError("budget_chars must be non-negative")
        selected = list(pinned)
        if max(
            len(cls.render_record_set(selected)),
            len(cls.render_record_set(selected, structured=True)),
        ) > budget_chars:
            raise ValueError("budget is too small for pinned instructions and active request")
        for candidate in candidates:
            proposed = selected + [candidate]
            if max(
                len(cls.render_record_set(proposed)),
                len(cls.render_record_set(proposed, structured=True)),
            ) <= budget_chars:
                selected = proposed
        return (
            cls.render_record_set(selected),
            cls.render_record_set(selected, structured=True),
            selected,
        )

    def pinned_records(self, compiled: CompiledConversation) -> list[Record]:
        """System records and the active user request, in source order."""
        records = self._public_records(compiled)
        roles = {turn["id"]: turn["role"] for turn in compiled.turns}
        latest_user = next(
            (turn["id"] for turn in reversed(compiled.turns) if turn["role"] == "user"), None
        )
        return [
            record for record in records
            if latest_user in record["source_ids"]
            or any(roles[source] == "system" for source in record["source_ids"])
        ]

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
        records = self._public_records(compiled)
        role_by_source = {turn["id"]: turn["role"] for turn in compiled.turns}
        latest_user = next(
            (turn["id"] for turn in reversed(compiled.turns) if turn["role"] == "user"), None
        )
        pinned = [
            record
            for record in records
            if latest_user in record["source_ids"]
            or any(role_by_source[source] == "system" for source in record["source_ids"])
        ]
        pinned_ids = {record["id"] for record in pinned}
        candidates = [record for record in records if record["id"] not in pinned_ids]
        return self._fit_sections(pinned, candidates, budget_chars)

    def render_organized(self, organized: Mapping[str, Any], *, budget_chars: int) -> str:
        """Render attention-ranked clusters under an exact character ceiling."""
        candidates = [
            record
            for cluster in organized.get("clusters", [])
            for record in cluster.get("records", [])
        ]
        return self._fit_sections(organized.get("pinned", []), candidates, budget_chars)
