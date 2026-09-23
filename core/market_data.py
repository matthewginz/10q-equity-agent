"""
Real price/valuation context -- fed into the pipeline's final synthesis
step so it can talk about where a stock's sentiment sits right now and why
it trades at its current price, not just what the filing's own fundamentals
say in isolation.

Deliberately point-in-time capable: the live app wants TODAY's price, but
the backtest harness (scripts/backtest_accuracy.py) needs the price AS OF
THE FILING DATE it's scoring -- feeding it today's price for a filing from
months ago would leak future information into the "why is it at this
price" narrative, the same look-ahead-bias problem facts_as_of() guards
against for XBRL data.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

from core._net import ensure_curl_cffi_ca_bundle

ensure_curl_cffi_ca_bundle()

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketSnapshot:
    as_of: str
    price: float
    fifty_two_week_low: float
    fifty_two_week_high: float
    return_90d_pct: float | None
    return_365d_pct: float | None


# Found live: concurrent yfinance calls from several threads (the backtest's
# workers; the app's parallel compare mode) can come back empty. One lock.
_yf_lock = threading.Lock()


def fetch_market_snapshot(ticker: str, as_of_date: str | None = None) -> MarketSnapshot | None:
    """Price context as of as_of_date (a past filing date), or as of now
    when as_of_date is None. None if yfinance has no data in that window
    (e.g. delisted since, or a temporary outage) -- callers must treat a
    missing snapshot as "no market context available", never fabricate one."""
    as_of = datetime.fromisoformat(as_of_date) if as_of_date else datetime.utcnow()
    window_start = as_of - timedelta(days=370)
    with _yf_lock:
        hist = yf.Ticker(ticker).history(
            start=window_start.strftime("%Y-%m-%d"),
            end=(as_of + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
    if hist.empty:
        return None
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    return snapshot_from_closes(hist["Close"], as_of)


def snapshot_from_closes(closes: pd.Series, as_of: datetime) -> MarketSnapshot | None:
    """Pure part of fetch_market_snapshot, split out to be unit-testable.

    Found live: yfinance returns a row for the current day whose Close is NaN
    until the session's data settles. Taking closes.iloc[-1] blindly made the
    "current price" NaN -- the UI rendered "$nan" AND the pipeline's final
    step was fed "Price: $nan". Blank rows are dropped first; the latest real
    close is the price."""
    closes = closes.dropna()
    closes = closes[closes > 0]
    if closes.empty:
        return None
    price = float(closes.iloc[-1])

    def _return_over(days: int) -> float | None:
        earlier = closes.index[closes.index <= as_of - timedelta(days=days)]
        if len(earlier) == 0:
            return None
        base = float(closes.loc[earlier[-1]])
        return (price - base) / base

    return MarketSnapshot(
        as_of=closes.index[-1].strftime("%Y-%m-%d"),
        price=price,
        fifty_two_week_low=float(closes.min()),
        fifty_two_week_high=float(closes.max()),
        return_90d_pct=_return_over(90),
        return_365d_pct=_return_over(365),
    )


def format_market_snapshot(snapshot: MarketSnapshot | None) -> str:
    if snapshot is None:
        return "Current market price data was not available for this run."
    parts = [
        f"Price: ${snapshot.price:,.2f} (as of {snapshot.as_of})",
        f"52-week range: ${snapshot.fifty_two_week_low:,.2f} - ${snapshot.fifty_two_week_high:,.2f}",
    ]
    if snapshot.return_90d_pct is not None:
        parts.append(f"Trailing 90-day return: {snapshot.return_90d_pct:+.1%}")
    if snapshot.return_365d_pct is not None:
        parts.append(f"Trailing 365-day return: {snapshot.return_365d_pct:+.1%}")
    return "\n".join(parts)
