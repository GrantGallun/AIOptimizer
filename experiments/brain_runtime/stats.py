#!/usr/bin/env python3
"""Small statistics helpers for reading experiment gates honestly.

Added after the 2026-07-11 self-audit: every gate so far compared point estimates
(e.g. 0.872 vs 1.000 on n=120) with no interval, and hidden seeds 101/103/107 were
reused across pre-registrations until they stopped being hidden. This module gives
gates a conservative significance check and a registry of fresh hidden seeds.
"""

from __future__ import annotations

import math

# Fresh hidden seeds for FUTURE confirmatory reads. 101/103/107 are burned — they were
# read (and design decisions made) in v3.1/v4/v4.2/v5. Claim a tuple per prereg version
# and never reuse one after its first read.
FRESH_HIDDEN_SEEDS: dict[str, tuple[int, int, int]] = {
    "v6": (211, 223, 227),
    "v7": (307, 311, 313),
    "v8": (401, 409, 419),
    "v9": (503, 509, 521),
    "v10": (601, 607, 613),
    "v11": (701, 709, 719),
    "v12": (809, 811, 821),
    "v13": (1009, 1013, 1019),
    "v14": (1109, 1117, 1123),
}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def gap_significant(s1: int, n1: int, s2: int, n2: int, z: float = 1.96) -> bool:
    """Conservative check: True iff the two Wilson intervals do not overlap."""
    low1, high1 = wilson_interval(s1, n1, z)
    low2, high2 = wilson_interval(s2, n2, z)
    return low1 > high2 or low2 > high1
