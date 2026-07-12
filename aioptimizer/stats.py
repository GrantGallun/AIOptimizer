"""Small public statistics helpers for optimization receipts."""

from __future__ import annotations

import math


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
    """Return whether two Wilson intervals do not overlap."""
    low1, high1 = wilson_interval(s1, n1, z)
    low2, high2 = wilson_interval(s2, n2, z)
    return low1 > high2 or low2 > high1


__all__ = ["gap_significant", "wilson_interval"]
