"""
Agent vs. Wall Street: puts each backtested stance (docs/backtest/results.csv,
written by backtest_accuracy.py) next to the sell-side consensus AS OF THAT
SAME FILING DATE, rebuilt from yfinance's dated rating history (see
core/analyst_consensus.py for the method).

No LLM calls -- this only reads the existing backtest results plus free
rating/price history, so it's cheap to rerun.

Output: docs/backtest/analyst_comparison.csv (one row per backtested
filing), read by the app's Track Record page, plus a printed summary.

Usage:
    python scripts/analyst_comparison.py
    python scripts/analyst_comparison.py --tickers BA   # smoke test one
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core._net import ensure_curl_cffi_ca_bundle

ensure_curl_cffi_ca_bundle()

import pandas as pd
import yfinance as yf

from core import market_data
from core.analyst_consensus import consensus_as_of

BACKTEST_DIR = Path(__file__).resolve().parent.parent / "docs" / "backtest"
RESULTS_CSV = BACKTEST_DIR / "results.csv"
OUT_CSV = BACKTEST_DIR / "analyst_comparison.csv"
INTER_TICKER_DELAY_SECONDS = 0.5


@dataclass(frozen=True)
class ComparisonRow:
    ticker: str
    company: str
    filing_date: str
    agent_stance: str
    street_stance: str
    n_firms: int
    n_buy: int
    n_hold: int
    n_sell: int
    street_score: float | None
    price_at_filing: float | None
    mean_price_target: float | None
    implied_upside_pct: float | None
    actual_return_pct: float | None
    excess_return_pct: float | None  # vs SPY over the same window
    agent_hit: bool | None
    street_hit: bool | None
    agent_hit_vs_market: bool | None
    street_hit_vs_market: bool | None
    agrees: bool | None  # None = no usable Street consensus to compare against
    model: str


def directional_hit(stance: str, return_pct: float | None) -> bool | None:
    if return_pct is None or stance not in ("Bullish", "Bearish"):
        return None
    return (stance == "Bullish") == (return_pct > 0)


def _float_or_none(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


_ratings_cache: dict[str, pd.DataFrame] = {}


def _ratings(ticker: str) -> pd.DataFrame:
    """One rating-history download per ticker, reused across its filings.
    An empty frame (no history) becomes an 'Insufficient' consensus row
    rather than dropping the filing from the comparison."""
    if ticker not in _ratings_cache:
        ratings = yf.Ticker(ticker).upgrades_downgrades
        _ratings_cache[ticker] = ratings if ratings is not None else pd.DataFrame()
    return _ratings_cache[ticker]


def compare_row(result: dict) -> ComparisonRow:
    ticker, filing_date = result["ticker"], result["filing_date"]
    snap = consensus_as_of(_ratings(ticker), filing_date)
    market = market_data.fetch_market_snapshot(ticker, as_of_date=filing_date)
    price = market.price if market else None
    upside = (
        round((snap.mean_price_target / price - 1) * 100, 2)
        if price and snap.mean_price_target else None
    )
    actual = _float_or_none(result["actual_return_pct"])
    excess = _float_or_none(result.get("excess_return_pct", ""))
    return ComparisonRow(
        ticker=ticker, company=result["company"], filing_date=filing_date,
        agent_stance=result["stance"], street_stance=snap.stance,
        n_firms=snap.n_firms, n_buy=snap.n_buy, n_hold=snap.n_hold, n_sell=snap.n_sell,
        street_score=snap.score,
        price_at_filing=round(price, 2) if price else None,
        mean_price_target=snap.mean_price_target,
        implied_upside_pct=upside,
        actual_return_pct=actual,
        excess_return_pct=excess,
        agent_hit=directional_hit(result["stance"], actual),
        street_hit=directional_hit(snap.stance, actual),
        agent_hit_vs_market=directional_hit(result["stance"], excess),
        street_hit_vs_market=directional_hit(snap.stance, excess),
        agrees=None if snap.stance == "Insufficient" else result["stance"] == snap.stance,
        model=result.get("models", ""),
    )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[ComparisonRow]) -> None:
    def hit_rate(hits: list[bool | None]) -> str:
        scored = [h for h in hits if h is not None]
        return f"{sum(scored)}/{len(scored)}" if scored else "n/a"

    comparable = [r for r in rows if r.agrees is not None]
    agree = [r for r in comparable if r.agrees]
    disagree = [r for r in comparable if not r.agrees]
    print("\n" + "=" * 60)
    print(f"Filings: {len(rows)} ({len(rows) - len(comparable)} with no usable Street rating history)")
    print(f"Agent agrees with Street consensus: {len(agree)}/{len(comparable)}")
    print(f"Agent directional hit rate:  {hit_rate([r.agent_hit for r in rows])}")
    print(f"Street directional hit rate: {hit_rate([r.street_hit for r in rows])}")
    for label, group in (("agreed", agree), ("disagreed", disagree)):
        avg = _mean([r.actual_return_pct for r in group if r.actual_return_pct is not None])
        print(f"Avg 90d return when agent {label}: "
              f"{'n/a' if avg is None else f'{avg:+.2f}%'} (n={len(group)})")
    for r in disagree:
        print(f"  {r.ticker}: agent {r.agent_stance} vs Street {r.street_stance} "
              f"-> {r.actual_return_pct:+.2f}%")
    print("=" * 60)


def main(tickers: list[str] | None) -> None:
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        # Only rows from the current harness (model recorded, one model per filing).
        results = [r for r in csv.DictReader(f)
                   if r.get("models") and (not tickers or r["ticker"] in tickers)]

    rows: list[ComparisonRow] = []
    for result in results:
        print(f"=== {result['ticker']} ({result['filing_date']}) ===")
        try:
            row = compare_row(result)
        except Exception as exc:  # noqa: BLE001 -- one ticker's failure shouldn't kill the run
            print(f"  FAILED: {exc}")
            continue
        print(f"  agent={row.agent_stance} street={row.street_stance} "
              f"({row.n_buy}B/{row.n_hold}H/{row.n_sell}S of {row.n_firms}) "
              f"upside={row.implied_upside_pct}%")
        rows.append(row)
        time.sleep(INTER_TICKER_DELAY_SECONDS)

    if rows and not tickers:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in rows)
        print(f"\nWrote {OUT_CSV}")
    summarize(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Only these tickers; prints results without writing the CSV.")
    main(parser.parse_args().tickers)
