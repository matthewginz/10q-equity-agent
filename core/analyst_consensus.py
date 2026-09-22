"""
Point-in-time Wall Street consensus, rebuilt from yfinance's dated
analyst rating history (`Ticker.upgrades_downgrades`).

Used by scripts/analyst_comparison.py to put the agent's backtested
Bullish/Neutral/Bearish stance next to what the sell side was saying on the
SAME date -- not today's consensus, which would already know how the
following quarter played out (same look-ahead-bias rule as
backtest_accuracy.facts_as_of and market_data's as_of_date).

Method: for each firm, keep only its most recent rating dated strictly
before the filing date and within LOOKBACK_DAYS (older ratings are treated
as dropped coverage). Map each rating to +1 / 0 / -1, average across
firms, and bucket that average into a stance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

LOOKBACK_DAYS = 365
MIN_FIRMS = 3                 # fewer covering firms than this isn't a "consensus"
STANCE_THRESHOLD = 1 / 3      # mean score beyond +/- this leans Bullish/Bearish

# Normalized (lowercased, spaces/hyphens stripped) broker grade -> score.
# Grades not listed here are counted as unmapped rather than guessed.
_GRADE_SCORES: dict[str, int] = {
    **dict.fromkeys(
        ["buy", "strongbuy", "overweight", "outperform", "positive", "accumulate",
         "marketoutperform", "sectoroutperform", "add", "toppick", "convictionbuy",
         "longtermbuy", "speculativebuy", "moderatebuy"], 1),
    **dict.fromkeys(
        ["hold", "neutral", "equalweight", "marketperform", "sectorperform", "inline",
         "peerperform", "perform", "sectorweight", "fairvalue", "marketweight"], 0),
    **dict.fromkeys(
        ["sell", "strongsell", "underweight", "underperform", "negative", "reduce",
         "marketunderperform", "sectorunderperform", "moderatesell"], -1),
}


@dataclass(frozen=True)
class ConsensusSnapshot:
    as_of: str
    n_firms: int
    n_buy: int
    n_hold: int
    n_sell: int
    n_unmapped: int
    score: float | None           # mean of +1/0/-1 across mapped firms
    stance: str                   # Bullish / Neutral / Bearish / Insufficient
    mean_price_target: float | None


def grade_score(grade: str | None) -> int | None:
    if not grade:
        return None
    key = grade.lower().replace(" ", "").replace("-", "")
    return _GRADE_SCORES.get(key)


def stance_from_score(score: float | None, n_firms: int) -> str:
    if score is None or n_firms < MIN_FIRMS:
        return "Insufficient"
    if score > STANCE_THRESHOLD:
        return "Bullish"
    if score < -STANCE_THRESHOLD:
        return "Bearish"
    return "Neutral"


def consensus_as_of(ratings: pd.DataFrame, as_of_date: str,
                    lookback_days: int = LOOKBACK_DAYS) -> ConsensusSnapshot:
    """`ratings` is the upgrades_downgrades frame: GradeDate index, with
    Firm / ToGrade / currentPriceTarget columns. Pure function -- no
    network -- so it's unit-testable against a synthetic frame."""
    cutoff = datetime.fromisoformat(as_of_date)
    window_start = cutoff - timedelta(days=lookback_days)

    frame = ratings
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame = frame.tz_localize(None)
    frame = frame[(frame.index < cutoff) & (frame.index >= window_start)]
    latest = frame.sort_index().groupby("Firm").tail(1)

    scores = [grade_score(g) for g in latest.get("ToGrade", pd.Series(dtype=str))]
    mapped = [s for s in scores if s is not None]
    n_firms = len(mapped)
    score = sum(mapped) / n_firms if n_firms else None

    targets = latest.get("currentPriceTarget", pd.Series(dtype=float))
    targets = targets[targets > 0] if len(targets) else targets
    mean_target = float(targets.mean()) if len(targets) else None

    return ConsensusSnapshot(
        as_of=as_of_date,
        n_firms=n_firms,
        n_buy=mapped.count(1),
        n_hold=mapped.count(0),
        n_sell=mapped.count(-1),
        n_unmapped=len(scores) - n_firms,
        score=round(score, 3) if score is not None else None,
        stance=stance_from_score(score, n_firms),
        mean_price_target=round(mean_target, 2) if mean_target is not None else None,
    )
