import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import backtest_accuracy as bt
from core.edgar_client import FilingRef


def _case(ticker: str, report_date: str, filing_date: str) -> bt.Case:
    return bt.Case(ticker, FilingRef("0-0-0", filing_date, report_date, "doc.htm", ticker, 1))


def _cases() -> list[bt.Case]:
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "KO", "PG", "GE", "T"]
    quarters = [("2025-03-31", "2025-05-01"), ("2025-06-30", "2025-08-01"), ("2024-09-30", "2024-11-01")]
    return [_case(t, r, f) for r, f in quarters for t in tickers]


def test_newest_quarter_comes_first():
    ordered = bt.order_newest_first(_cases())
    report_dates = [c.filing.report_date for c in ordered]
    assert report_dates[:8] == ["2025-06-30"] * 8
    assert report_dates[8:16] == ["2025-03-31"] * 8
    assert report_dates[16:] == ["2024-09-30"] * 8


def test_quarter_is_calendar_not_exact_date():
    # retail fiscal quarters end off-calendar; Jun 28 and Jun 30 are the same quarter
    ordered = bt.order_newest_first([
        _case("AAPL", "2025-06-28", "2025-08-01"),
        _case("KO", "2025-03-28", "2025-04-29"),
        _case("MSFT", "2025-06-30", "2025-07-30"),
    ])
    assert {c.ticker for c in ordered[:2]} == {"AAPL", "MSFT"}
    assert ordered[2].ticker == "KO"


def test_within_quarter_order_is_shuffled_and_reproducible():
    first = [c.ticker for c in bt.order_newest_first(_cases())][:8]
    again = [c.ticker for c in bt.order_newest_first(list(reversed(_cases())))][:8]
    assert first == again                  # same input set -> same order, whatever order it arrived in
    assert first != sorted(first)          # not alphabetical: a partial quarter is a random slice


def test_missing_report_date_falls_back_to_filing_date():
    ordered = bt.order_newest_first([_case("OLD", "", "2024-11-01"), _case("NEW", "", "2025-08-01")])
    assert [c.ticker for c in ordered] == ["NEW", "OLD"]


def test_universe_is_the_frozen_sp500_list():
    assert len(bt.TICKERS) > 490
    assert len(set(bt.TICKERS)) == len(bt.TICKERS)
    assert {"AAPL", "JPM", "BRK-B"} <= set(bt.TICKERS)
