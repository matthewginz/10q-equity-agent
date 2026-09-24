import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.benchmarks import baselines, long_short


def _frame(rows: list[tuple[str, str, float | None, float | None]]) -> pd.DataFrame:
    """(quarter, agent_stance, return %, prior 90-day return %)"""
    return pd.DataFrame(rows, columns=["quarter", "agent_stance", "ret", "prior_90d_return_pct"])


def test_always_bullish_scores_every_filing_with_a_return():
    df = _frame([("2025 Q1", "Neutral", 5.0, 1.0), ("2025 Q1", "Bearish", -2.0, 1.0),
                 ("2025 Q1", "Bullish", 3.0, None), ("2025 Q1", "Bullish", None, 1.0)])
    always = {b.name: b for b in baselines(df, "ret")}["Always Bullish"]
    assert (always.rate.hits, always.rate.n) == (2, 3)   # the row with no return isn't scored


def test_momentum_calls_last_quarters_direction():
    df = _frame([("2025 Q1", "Neutral", 5.0, 10.0),    # up before, up after: hit
                 ("2025 Q1", "Neutral", 4.0, -3.0),    # down before, up after: miss
                 ("2025 Q1", "Neutral", -6.0, -1.0),   # down before, down after: hit
                 ("2025 Q1", "Neutral", 2.0, None)])   # no prior return: not scored
    momentum = {b.name: b for b in baselines(df, "ret")}["Momentum"]
    assert (momentum.rate.hits, momentum.rate.n) == (2, 3)


def test_long_short_spread_per_quarter_and_overall():
    df = _frame([
        ("2025 Q1", "Bullish", 10.0, None), ("2025 Q1", "Bullish", 6.0, None), ("2025 Q1", "Bullish", 8.0, None),
        ("2025 Q1", "Bearish", 1.0, None), ("2025 Q1", "Bearish", 3.0, None), ("2025 Q1", "Bearish", 2.0, None),
        ("2025 Q2", "Bullish", -1.0, None), ("2025 Q2", "Bullish", 1.0, None), ("2025 Q2", "Bullish", 0.0, None),
        ("2025 Q2", "Bearish", 4.0, None), ("2025 Q2", "Bearish", 2.0, None), ("2025 Q2", "Bearish", 3.0, None),
        ("2025 Q2", "Neutral", 50.0, None),   # neutral calls are never traded
    ])
    result = long_short(df, "ret", min_per_side=3)
    assert list(result.cohorts["quarter"]) == ["2025 Q1", "2025 Q2"]
    assert list(result.cohorts["spread"]) == pytest.approx([6.0, -3.0])
    assert result.mean_spread == pytest.approx(1.5)          # every Bullish avg 4.0 minus every Bearish avg 2.5
    assert result.ci_low < result.mean_spread < result.ci_high
    assert result.winning_cohorts == 1
    assert result.cumulative == pytest.approx([1.06, 1.06 * 0.97])
    assert result.max_drawdown == pytest.approx(-0.03)
    assert result.sharpe is None                             # 2 quarters is too few to annualize honestly


def test_thin_quarters_are_left_out_of_the_portfolio():
    df = _frame([("2025 Q1", "Bullish", 5.0, None), ("2025 Q1", "Bearish", 1.0, None)])
    result = long_short(df, "ret", min_per_side=3)
    assert result.cohorts.empty
    assert result.sharpe is None
