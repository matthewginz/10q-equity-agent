"""
The two questions a hit rate alone can't answer, for the Track Record page:

1. Baselines -- is the agent better than a caller with no brains at all?
   "Always Bullish" wins most of the time in a rising market, and
   "Momentum" (call the direction of the stock's prior 90 days, as of the
   filing date) is the classic free signal. The agent has to beat these,
   not just a coin flip.

2. Long-short portfolio -- would the calls have made money? Each filing
   quarter is one cohort: equal-weight long every Bullish call, short every
   Bearish call, each held the same 90 days the hit rate is scored on.
   Neutral calls are never traded. No transaction costs or borrow fees.

Pure pandas, no network: everything comes from docs/backtest/analyst_comparison.csv.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.stats import HitRate, hit_rate

COHORTS_PER_YEAR = 4          # 90-day holds, one cohort per filing quarter
BOOTSTRAP_REPS = 10_000
MIN_PER_SIDE = 5              # a quarter needs this many longs AND shorts to count
MIN_COHORTS_FOR_SHARPE = 4    # below this a Sharpe is one lucky quarter, not a ratio worth showing


@dataclass(frozen=True)
class Baseline:
    name: str
    description: str
    rate: HitRate


def _hit_rate(calls: pd.Series, returns: pd.Series) -> HitRate:
    """Directional hits of Bullish/Bearish `calls` against `returns` (%),
    skipping rows where either is missing."""
    scored = calls.isin(["Bullish", "Bearish"]) & returns.notna()
    hits = (calls[scored] == "Bullish") == (returns[scored] > 0)
    return hit_rate(int(hits.sum()), int(scored.sum()))


def baselines(df: pd.DataFrame, ret_col: str) -> list[Baseline]:
    always = pd.Series("Bullish", index=df.index)
    prior = df["prior_90d_return_pct"]
    momentum = prior.map(lambda p: None if pd.isna(p) else ("Bullish" if p > 0 else "Bearish"))
    return [
        Baseline("Always Bullish", "Calls every filing Bullish.", _hit_rate(always, df[ret_col])),
        Baseline("Momentum", "Calls the direction of the stock's prior 90 days, as of the filing date.",
                 _hit_rate(momentum, df[ret_col])),
    ]


@dataclass(frozen=True)
class LongShort:
    cohorts: pd.DataFrame          # quarter, n_long, n_short, long_ret, short_ret, spread (all % per 90 days)
    mean_spread: float | None      # every Bullish avg minus every Bearish avg (% per 90 days)
    ci_low: float | None           # quarter-clustered 95% interval on that difference (None under 2 quarters)
    ci_high: float | None
    sharpe: float | None           # annualized, from the quarterly cohort spreads
    winning_cohorts: int
    cumulative: list[float]        # growth of $1 run through the cohorts in order
    max_drawdown: float | None     # worst peak-to-trough of `cumulative` (e.g. -0.08)


def cluster_ci(per_cluster: pd.DataFrame, stat, reps: int = BOOTSTRAP_REPS,
               seed: int = 0) -> tuple[float | None, float | None]:
    """95% bootstrap interval for stat(column totals), resampling whole filing quarters.
    Filings from the same quarter share one market, so they aren't independent; treating
    them as if they were (a Welch or McNemar test) overstates significance."""
    k = len(per_cluster)
    if k < 2:
        return None, None
    counts = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=reps)
    totals = pd.DataFrame(counts @ per_cluster.to_numpy(dtype=float), columns=per_cluster.columns)
    draws = stat(totals).replace([np.inf, -np.inf], np.nan).dropna()   # resamples missing one side
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _spread_ci(traded: pd.DataFrame, ret_col: str) -> tuple[float | None, float | None, float | None]:
    long = traded.loc[traded["agent_stance"] == "Bullish", ret_col]
    short = traded.loc[traded["agent_stance"] == "Bearish", ret_col]
    if len(long) < 2 or len(short) < 2:
        return None, None, None
    is_long = traded["agent_stance"] == "Bullish"
    per_quarter = pd.DataFrame({
        "long_sum": traded[ret_col].where(is_long, 0), "n_long": is_long.astype(int),
        "short_sum": traded[ret_col].where(~is_long, 0), "n_short": (~is_long).astype(int),
    }).groupby(traded["quarter"]).sum()
    low, high = cluster_ci(per_quarter, lambda t: t.long_sum / t.n_long - t.short_sum / t.n_short)
    return float(long.mean() - short.mean()), low, high


@dataclass(frozen=True)
class HeadToHead:
    agent: float | None       # agent hit rate on filings where both made a scored directional call
    street: float | None
    n: int
    ci_low: float | None      # quarter-clustered 95% interval on agent minus Street (pts)
    ci_high: float | None


def head_to_head(df: pd.DataFrame, agent_col: str, street_col: str) -> HeadToHead:
    both = df[df[agent_col].notna() & df[street_col].notna()]
    if both.empty:
        return HeadToHead(None, None, 0, None, None)
    agent, street = both[agent_col].astype(bool), both[street_col].astype(bool)
    per_quarter = pd.DataFrame({"agent": agent.astype(int), "street": street.astype(int), "n": 1}
                               ).groupby(both["quarter"]).sum()
    low, high = cluster_ci(per_quarter, lambda t: 100 * (t.agent - t.street) / t.n)
    return HeadToHead(float(agent.mean()), float(street.mean()), len(both), low, high)


def _max_drawdown(curve: list[float]) -> float | None:
    if not curve:
        return None
    peak, worst = 1.0, 0.0
    for value in curve:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst


def long_short(df: pd.DataFrame, ret_col: str, min_per_side: int = MIN_PER_SIDE) -> LongShort:
    traded = df[df["agent_stance"].isin(["Bullish", "Bearish"]) & df[ret_col].notna()]
    rows = []
    for quarter, group in traded.groupby("quarter", sort=True):
        long = group.loc[group["agent_stance"] == "Bullish", ret_col]
        short = group.loc[group["agent_stance"] == "Bearish", ret_col]
        if len(long) >= min_per_side and len(short) >= min_per_side:
            rows.append({"quarter": quarter, "n_long": len(long), "n_short": len(short),
                         "long_ret": long.mean(), "short_ret": short.mean(),
                         "spread": long.mean() - short.mean()})
    cohorts = pd.DataFrame(rows, columns=["quarter", "n_long", "n_short", "long_ret", "short_ret", "spread"])

    spreads = cohorts["spread"]
    sharpe = (float(spreads.mean() / spreads.std() * math.sqrt(COHORTS_PER_YEAR))
              if len(spreads) >= MIN_COHORTS_FOR_SHARPE and spreads.std() > 0 else None)
    cumulative = (1 + spreads / 100).cumprod().tolist()
    mean, low, high = _spread_ci(traded, ret_col)
    return LongShort(cohorts=cohorts, mean_spread=mean, ci_low=low, ci_high=high, sharpe=sharpe,
                     winning_cohorts=int((spreads > 0).sum()), cumulative=cumulative,
                     max_drawdown=_max_drawdown(cumulative))
