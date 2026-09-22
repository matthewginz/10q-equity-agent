import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyst_consensus import consensus_as_of, grade_score, stance_from_score


def _ratings(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["GradeDate", "Firm", "ToGrade", "currentPriceTarget"])
    frame["GradeDate"] = pd.to_datetime(frame["GradeDate"], format="ISO8601")
    return frame.set_index("GradeDate")


def test_grade_score_normalizes_broker_wording():
    assert grade_score("Equal-Weight") == 0
    assert grade_score("Strong Buy") == 1
    assert grade_score("Sector Underperform") == -1
    assert grade_score("Some New Grade") is None
    assert grade_score(None) is None


def test_uses_each_firms_latest_rating_before_the_date():
    ratings = _ratings([
        ("2025-01-10", "A", "Sell", 80.0),
        ("2025-06-01", "A", "Buy", 120.0),   # A's latest before cutoff -> Buy
        ("2025-05-01", "B", "Buy", 110.0),
        ("2025-05-02", "C", "Overweight", 130.0),
        ("2025-07-02", "C", "Underweight", 50.0),  # after cutoff, ignored
    ])
    snap = consensus_as_of(ratings, "2025-07-01")
    assert (snap.n_buy, snap.n_hold, snap.n_sell) == (3, 0, 0)
    assert snap.stance == "Bullish"
    assert snap.mean_price_target == 120.0


def test_same_day_rating_is_excluded_as_look_ahead():
    ratings = _ratings([
        ("2025-06-01", "A", "Hold", 0.0),
        ("2025-06-01", "B", "Hold", 0.0),
        ("2025-06-01", "C", "Hold", 0.0),
        ("2025-07-01 09:00", "D", "Sell", 0.0),
    ])
    snap = consensus_as_of(ratings, "2025-07-01")
    assert snap.n_firms == 3
    assert snap.n_sell == 0
    assert snap.stance == "Neutral"
    assert snap.mean_price_target is None  # zero targets are "no target", not $0


def test_stale_ratings_outside_lookback_are_dropped():
    ratings = _ratings([
        ("2023-01-01", "A", "Buy", 100.0),
        ("2025-06-01", "B", "Buy", 100.0),
    ])
    snap = consensus_as_of(ratings, "2025-07-01")
    assert snap.n_firms == 1
    assert snap.stance == "Insufficient"


def test_unmapped_grades_are_counted_not_guessed():
    ratings = _ratings([
        ("2025-06-01", "A", "Buy", 0.0),
        ("2025-06-02", "B", "Mystery", 0.0),
    ])
    snap = consensus_as_of(ratings, "2025-07-01")
    assert snap.n_unmapped == 1
    assert snap.n_firms == 1


def test_no_rating_history_is_insufficient_not_an_error():
    snap = consensus_as_of(pd.DataFrame(), "2025-07-01")
    assert snap.stance == "Insufficient"
    assert snap.n_firms == 0


def test_stance_thresholds():
    assert stance_from_score(0.5, 10) == "Bullish"
    assert stance_from_score(0.2, 10) == "Neutral"
    assert stance_from_score(-0.5, 10) == "Bearish"
    assert stance_from_score(0.9, 2) == "Insufficient"
