"""
Small, dependency-free significance helpers for the Track Record page --
so a hit rate is always shown with how much it could be luck, not as a
bare percentage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963984540054   # two-sided 95% normal quantile


@dataclass(frozen=True)
class HitRate:
    hits: int
    n: int
    rate: float | None          # hits / n
    ci_low: float | None        # Wilson 95% interval
    ci_high: float | None
    p_value: float | None       # one-sided exact binomial, H0: rate = 0.5 (coin flip)


def wilson_interval(hits: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval: well-behaved at small n and near 0%/100%,
    unlike the textbook normal approximation (which can go below 0 or
    above 1 at the sample sizes this project actually has)."""
    if n == 0:
        return None
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def binomial_p_value(hits: int, n: int, p0: float = 0.5) -> float | None:
    """P(X >= hits) for X ~ Binomial(n, p0): how often a caller with no
    skill (right with probability p0) would do at least this well."""
    if n == 0:
        return None
    return sum(math.comb(n, k) * p0 ** k * (1 - p0) ** (n - k) for k in range(hits, n + 1))


def hit_rate(hits: int, n: int) -> HitRate:
    interval = wilson_interval(hits, n)
    return HitRate(
        hits=hits, n=n,
        rate=hits / n if n else None,
        ci_low=interval[0] if interval else None,
        ci_high=interval[1] if interval else None,
        p_value=binomial_p_value(hits, n),
    )
