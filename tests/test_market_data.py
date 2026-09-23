import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.market_data import format_market_snapshot, snapshot_from_closes


def _closes(values: list[float], end: str = "2026-09-22") -> pd.Series:
    index = pd.bdate_range(end=end, periods=len(values))
    return pd.Series(values, index=index, dtype=float)


def test_blank_close_for_today_is_ignored():
    # regression: yfinance's in-progress row for today has a NaN close -> "$nan" price
    snap = snapshot_from_closes(_closes([100.0, 101.0, 102.0, float("nan")]), datetime(2026, 9, 22))
    assert snap.price == pytest.approx(102.0)
    assert snap.as_of == "2026-09-21"
    assert math.isfinite(snap.fifty_two_week_high)
    assert "nan" not in format_market_snapshot(snap).lower()


def test_returns_use_real_closes():
    closes = _closes([50.0] * 70 + [100.0] * 30)
    snap = snapshot_from_closes(closes, closes.index[-1].to_pydatetime())
    assert snap.return_90d_pct == pytest.approx(1.0)   # 50 -> 100
    assert snap.return_365d_pct is None                # not enough history


def test_all_blank_means_no_snapshot():
    assert snapshot_from_closes(_closes([float("nan"), float("nan")]), datetime(2026, 9, 22)) is None
