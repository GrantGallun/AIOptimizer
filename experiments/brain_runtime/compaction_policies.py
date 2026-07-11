"""Model-free policies for selecting memories during compaction."""

from __future__ import annotations

import math
import random
from typing import TypeVar


T = TypeVar("T")


def _validate_keep(keep: int) -> None:
    if keep <= 0:
        raise ValueError("keep must be greater than zero")


def keep_newest(
    items: list[T], keep: int, *, clock: int, seed: int = 0
) -> list[T]:
    """Keep the most recently created items, ordered newest first.

    ``created_at`` is the creation-order field. If an item does not expose it,
    its zero-based position in ``items`` is used instead. Later input positions
    also break equal ``created_at`` values, so selection remains deterministic.
    ``clock`` and ``seed`` are accepted for a uniform policy interface.
    """

    _validate_keep(keep)
    indexed = list(enumerate(items))
    ranked = sorted(
        indexed,
        key=lambda pair: (getattr(pair[1], "created_at", pair[0]), pair[0]),
        reverse=True,
    )
    return [item for _, item in ranked[:keep]]


def keep_random(
    items: list[T], keep: int, *, clock: int, seed: int = 0
) -> list[T]:
    """Keep a reproducible random sample without changing global RNG state."""

    _validate_keep(keep)
    return random.Random(seed).sample(items, k=min(keep, len(items)))


def keep_actr(
    items: list[T], keep: int, *, clock: int, seed: int = 0
) -> list[T]:
    """Keep items with the highest frozen ACT-R-style activation score.

    The score is ``log(1 + uses) + 0.5 / (1 + clock - last_accessed)``.
    Equal scores are ordered by item ID, ascending. ``seed`` is accepted for
    the uniform policy interface but is intentionally unused.
    """

    _validate_keep(keep)

    def score(item: T) -> float:
        recency = 1 / (1 + clock - item.last_accessed)  # type: ignore[attr-defined]
        return math.log(1 + item.uses) + 0.5 * recency  # type: ignore[attr-defined]

    ranked = sorted(items, key=lambda item: (-score(item), item.id))  # type: ignore[attr-defined]
    return ranked[:keep]
