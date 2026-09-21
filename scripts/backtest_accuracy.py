"""
Backtested directional-accuracy check for the pipeline's final Bullish /
Neutral / Bearish stance.

For each ticker: pulls the 10-Q from QUARTERS_BACK quarters ago (not the
latest one -- "latest" has no future yet), runs the real production
pipeline against it using ONLY XBRL facts the company had actually filed
by that filing's own filing date (point-in-time -- see facts_as_of below),
then checks via free historical price data whether the stock actually
moved in the stated direction over the following HORIZON_DAYS.

TICKERS is a fixed, sector/cap-diversified sample chosen before any run,
so results can't be cherry-picked after seeing them. Every run (hit,
miss, unparseable, or missing price data) is logged and written out --
the headline "directional accuracy" number is hits / (Bullish+Bearish
calls with price data available); Neutral calls aren't directional and
are reported separately, not silently dropped from the sample.

Usage:
    pip install -r requirements-backtest.txt
    python scripts/backtest_accuracy.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core._net import ensure_curl_cffi_ca_bundle

ensure_curl_cffi_ca_bundle()

import yfinance as yf

from core import edgar_client, market_data, ratios
from services import pipeline

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",   # mega-cap tech
    "JPM", "BAC", "GS",                        # financials
    "JNJ", "PFE", "UNH",                       # healthcare
    "XOM", "CVX",                               # energy
    "HD", "MCD", "NKE",                         # consumer
    "CAT", "BA", "GE",                          # industrials
    "DIS", "T",                                 # media / telecom
]

QUARTERS_BACK = 2      # skip the latest 10-Q so ~6 months of forward price history exists
HORIZON_DAYS = 90       # calendar days after filing over which the actual move is measured
PRICE_WINDOW_PAD_DAYS = 15   # extra days fetched past the horizon to absorb weekends/holidays
INTER_TICKER_DELAY_SECONDS = 1.0
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "backtest"


def facts_as_of(company_facts_json: dict, as_of_date: str) -> dict:
    """Trim every XBRL fact to only those the company had FILED by
    as_of_date. Without this, extract_facts would pick up data reported in
    filings AFTER the one being backtested (e.g. read a later quarter's
    number while scoring a stance formed off an earlier filing) -- that's
    look-ahead bias, not a real backtest."""
    trimmed: dict = {"facts": {"us-gaap": {}}}
    us_gaap = company_facts_json.get("facts", {}).get("us-gaap", {})
    for tag, node in us_gaap.items():
        new_units = {}
        for unit, facts in node.get("units", {}).items():
            kept = [f for f in facts if f.get("filed", "9999-99-99") <= as_of_date]
            if kept:
                new_units[unit] = kept
        if new_units:
            trimmed["facts"]["us-gaap"][tag] = {"units": new_units}
    return trimmed


def parse_stance(recommendation: str) -> str:
    """First unambiguous Bullish/Neutral/Bearish mention in the model's own
    final-stance text. The pipeline's system prompt requires the model
    state one of these three words explicitly, so this reads that
    instruction directly rather than inferring sentiment."""
    head = recommendation[:300].lower()
    for word in ("bullish", "bearish", "neutral"):
        if word in head:
            return word.capitalize()
    return "Unparseable"


def price_return(ticker: str, start_date: str, horizon_days: int) -> float | None:
    """Actual close-to-close return from the first trading day on/after
    start_date to the first trading day on/after start_date + horizon_days.
    None if yfinance has no data in that window (e.g. delisted since)."""
    start = datetime.fromisoformat(start_date)
    end = start + timedelta(days=horizon_days + PRICE_WINDOW_PAD_DAYS)
    hist = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    if hist.empty:
        return None
    entry = hist["Close"].iloc[0]
    if entry == 0:
        return None
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    target = start + timedelta(days=horizon_days)
    later = hist.index[hist.index >= target]
    exit_price = hist["Close"].loc[later[0]] if len(later) else hist["Close"].iloc[-1]
    return (exit_price - entry) / entry


@dataclass
class BacktestRow:
    ticker: str
    company: str
    filing_date: str
    report_date: str
    stance: str
    actual_return_pct: float | None
    hit: bool | None  # None = not directional (Neutral/Unparseable) or no price data


def _process_ticker(ticker: str) -> tuple[BacktestRow | None, dict | None]:
    filing = edgar_client.nth_recent_10q(ticker, QUARTERS_BACK)
    if filing is None:
        print("  no 10-Q found at that offset, skipping")
        return None, None

    raw_facts = edgar_client.company_facts(filing.cik)
    extracted = ratios.extract_facts(facts_as_of(raw_facts, filing.filing_date))
    computed = ratios.compute_ratios(extracted)
    filing_text = edgar_client.fetch_filing_text(filing)
    # Point-in-time market snapshot AS OF the filing date being scored, not
    # today -- feeding today's price into a backtest of a months-old filing
    # would leak future information into the "why is it at this price"
    # narrative, same look-ahead-bias problem facts_as_of() guards against.
    snapshot = market_data.fetch_market_snapshot(ticker, as_of_date=filing.filing_date)

    analysis = pipeline.run_analysis(
        company=filing.company_name, ticker=ticker,
        report_date=filing.report_date, ratios=computed, filing_text=filing_text,
        market=snapshot,
    )
    stance = parse_stance(analysis.recommendation)
    ret = price_return(ticker, filing.filing_date, HORIZON_DAYS)

    hit = None
    if ret is not None and stance in ("Bullish", "Bearish"):
        hit = (stance == "Bullish" and ret > 0) or (stance == "Bearish" and ret < 0)

    row = BacktestRow(
        ticker=ticker, company=filing.company_name,
        filing_date=filing.filing_date, report_date=filing.report_date,
        stance=stance,
        actual_return_pct=round(ret * 100, 2) if ret is not None else None,
        hit=hit,
    )
    print(f"  filed {filing.filing_date} -> stance={stance}, "
          f"actual {HORIZON_DAYS}d return={row.actual_return_pct}%, hit={hit}")

    log_entry = {
        "ticker": ticker, "filing_date": filing.filing_date, "report_date": filing.report_date,
        "recommendation": analysis.recommendation,
        "steps": [{"title": s.title, "output": s.output} for s in analysis.steps],
    }
    return row, log_entry


def run(tickers: list[str]) -> list[BacktestRow]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[BacktestRow] = []
    raw_log: list[dict] = []

    for ticker in tickers:
        print(f"\n=== {ticker} ===")
        try:
            row, log_entry = _process_ticker(ticker)
        except Exception as exc:  # noqa: BLE001 -- one ticker's failure shouldn't kill the run
            print(f"  FAILED: {exc}")
            row, log_entry = None, None
        if row is not None:
            rows.append(row)
            raw_log.append(log_entry)
        time.sleep(INTER_TICKER_DELAY_SECONDS)

    (OUT_DIR / "raw_runs.json").write_text(json.dumps(raw_log, indent=2), encoding="utf-8")
    if rows:
        with (OUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))
    return rows


def summarize(rows: list[BacktestRow]) -> None:
    directional = [r for r in rows if r.hit is not None]
    hits = [r for r in directional if r.hit]
    neutral = [r for r in rows if r.stance == "Neutral"]
    unparseable = [r for r in rows if r.stance == "Unparseable"]
    no_price = [r for r in rows if r.actual_return_pct is None]

    print("\n" + "=" * 60)
    print(f"Total tickers run: {len(rows)}")
    print(f"Directional calls (Bullish/Bearish, with price data): {len(directional)}")
    if directional:
        print(f"Directional accuracy: {len(hits)}/{len(directional)} = {100 * len(hits) / len(directional):.1f}%")
    print(f"Neutral calls (excluded from directional accuracy): {len(neutral)}")
    print(f"Unparseable stances: {len(unparseable)}")
    print(f"Missing price data: {len(no_price)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", nargs="+", metavar="TICKER",
        help="Run only these tickers instead of the full fixed sample "
             "(smoke-test one before committing to the full run).",
    )
    args = parser.parse_args()

    results = run(args.tickers or TICKERS)
    summarize(results)
