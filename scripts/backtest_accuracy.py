"""
Backtested directional-accuracy check for the pipeline's final Bullish /
Neutral / Bearish stance -- at scale, across several models, resumable.

Universe: TICKERS, a fixed 66-name, 11-sector large-cap list chosen before
any run so results can't be cherry-picked. For each ticker, EVERY 10-Q
filed between FILED_FROM and (today - horizon) is a backtest case, so the
sample spans several quarters and market regimes, not one.

Per filing: runs the real production pipeline using ONLY XBRL facts filed
by that filing's date (facts_as_of) and the price as of that date, with the
model told the filing date is "today" (pipeline as_of_date). Then scores the
stance against the stock's actual HORIZON_DAYS return, both raw and in
excess of SPY over the same window (a rising market otherwise flatters any
Bullish-leaning caller).

Models: Gemini's free tier caps each model separately, so one worker thread
per model in MODELS pulls filings off a shared queue. Every step of a given
filing is answered by that one worker's model (never mixed within a
filing), and the model is recorded per row so accuracy can be checked per
model. A worker rides out short overloads / per-minute limits by waiting;
once its model stays blocked it retires for the day and hands its filing
back to the queue for the others.

Resumable: cases run in a fixed-seed shuffled order (any prefix is a random
sample of the whole), results are written after every filing, and finished
(ticker, filing_date) cases are skipped -- rerun daily until done.

Usage:
    pip install -r requirements-backtest.txt
    python scripts/backtest_accuracy.py --limit 1      # smoke test one case
    python scripts/backtest_accuracy.py                # everything (resumable)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core._net import ensure_curl_cffi_ca_bundle

ensure_curl_cffi_ca_bundle()

import pandas as pd
import yfinance as yf
from google.genai.errors import ClientError, ServerError

from core import edgar_client, market_data, ratios
from services import gemini_client, pipeline

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "CRM", "ADBE", "INTC", "CSCO", "AMD",  # tech
    "JPM", "BAC", "GS", "WFC", "MS", "C", "AXP", "BLK",                                            # financials
    "JNJ", "PFE", "UNH", "MRK", "ABBV", "LLY", "TMO", "CVS",                                        # healthcare
    "XOM", "CVX", "COP", "SLB",                                                                    # energy
    "HD", "MCD", "NKE", "WMT", "COST", "PG", "KO", "PEP", "SBUX", "TGT", "LOW",                    # consumer
    "CAT", "BA", "GE", "UPS", "HON", "LMT", "DE",                                                  # industrials
    "DIS", "T", "VZ", "NFLX", "CMCSA",                                                             # media / telecom
    "NEE", "DUK", "SO",                                                                            # utilities
    "PLD", "AMT",                                                                                  # real estate
    "LIN", "NEM", "FCX",                                                                           # materials
    "TSLA", "F", "GM",                                                                             # autos
]
BENCHMARK = "SPY"

# Full-size Gemini flash models the free tier actually serves (probed live
# 2026-09-22). Excluded: pro models (free-tier limit 0), omni (limit 0),
# 2.5-series (retired, 404), lite models (weaker tier), and *-latest aliases
# (their underlying model can change, so a row couldn't say what answered).
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    # Google's open Gemma 4 (26B MoE): same API, a far larger free daily quota
    # (24+ requests in one day without a 429 vs. Gemini's 20), and number-
    # grounded output in spot checks. Needs thinking_level=minimal -- see
    # gemini_client._no_thinking. The 31B variant was overloaded/slow when probed.
    "gemma-4-26b-a4b-it",
]
BACKTEST_KEY_ENV = "GEMINI_BACKTEST_API_KEY"
SAME_MODEL_RETRY_WAITS = (5.0, 15.0, 30.0, 65.0)  # rides out 503s and per-minute 429s (~2 min)
UNAVAILABLE_COOLDOWN_SECONDS = 180                # a worker's pause after its model stays overloaded
MAX_UNAVAILABLE_STRIKES = 3                       # ...and how many of those before it retires

FILED_FROM = "2024-07-01"
HORIZON_DAYS = 90             # calendar days after filing over which the actual move is measured
HORIZON_SLACK_DAYS = 7        # a filing needs horizon + slack of real price history to be scored
MAX_10QS_PER_TICKER = 12      # enough to cover FILED_FROM for every name
SHUFFLE_SEED = 42
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "backtest"
RESULTS_CSV = OUT_DIR / "results.csv"
RAW_LOG = OUT_DIR / "raw_runs.json"
PROGRESS_JSON = OUT_DIR / "progress.json"   # read by the Track Record page


class QuotaExhausted(RuntimeError):
    """The worker's model is still rate-limited after every same-model retry."""


class ModelUnavailable(RuntimeError):
    """The worker's model is still overloaded (5xx) after every same-model retry."""


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


# ── Prices (one history download per ticker, reused across its filings) ──
# Found live: concurrent yfinance calls from the worker threads sometimes
# come back empty, and caching that empty result silently blanked a row's
# return. So every yfinance call goes through one lock, empty downloads are
# retried (never cached), and a still-missing return fails the case.
_yf_lock = threading.Lock()
_price_cache: dict[str, pd.Series] = {}
YF_RETRY_WAITS = (2.0, 5.0, 10.0)


def _download_closes(ticker: str) -> pd.Series:
    start = (datetime.fromisoformat(FILED_FROM) - timedelta(days=10)).strftime("%Y-%m-%d")
    for wait in (*YF_RETRY_WAITS, None):
        hist = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if not hist.empty:
            closes = hist["Close"].dropna()   # today's in-progress row has a NaN close
            if closes.index.tz is not None:
                closes.index = closes.index.tz_localize(None)
            return closes
        if wait is None:
            raise RuntimeError(f"no price history from yfinance for {ticker}")
        time.sleep(wait)
    raise AssertionError("unreachable")


def _closes(ticker: str) -> pd.Series:
    with _yf_lock:
        if ticker not in _price_cache:
            _price_cache[ticker] = _download_closes(ticker)
        return _price_cache[ticker]


def price_return(ticker: str, start_date: str, horizon_days: int) -> float | None:
    """Close-to-close total return (dividend-adjusted) from the first
    trading day on/after start_date to the first on/after start_date +
    horizon_days. None if that exit day hasn't happened yet or data is
    missing -- never a silently shortened horizon."""
    closes = _closes(ticker)
    start = datetime.fromisoformat(start_date)
    entry_idx = closes.index[closes.index >= start]
    exit_idx = closes.index[closes.index >= start + timedelta(days=horizon_days)]
    if not len(entry_idx) or not len(exit_idx):
        return None
    entry, exit_price = float(closes.loc[entry_idx[0]]), float(closes.loc[exit_idx[0]])
    return (exit_price - entry) / entry if entry else None


# ── Cases ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Case:
    ticker: str
    filing: edgar_client.FilingRef


@dataclass(frozen=True)
class BacktestRow:
    ticker: str
    company: str
    filing_date: str
    report_date: str
    stance: str
    actual_return_pct: float | None
    spy_return_pct: float | None
    excess_return_pct: float | None
    hit: bool | None            # None = not directional (Neutral/Unparseable) or no price data
    hit_vs_market: bool | None  # same, judged on return in excess of SPY
    models: str                 # the one model that answered every step of this filing


def _directional_hit(stance: str, value: float | None) -> bool | None:
    if value is None or stance not in ("Bullish", "Bearish"):
        return None
    return (stance == "Bullish") == (value > 0)


def _pct(value: float | None) -> float | None:
    return round(value * 100, 2) if value is not None else None


def build_worklist(tickers: list[str]) -> list[Case]:
    cutoff = (date.today() - timedelta(days=HORIZON_DAYS + HORIZON_SLACK_DAYS)).isoformat()
    cases: list[Case] = []
    for ticker in tickers:
        # Deliberately NOT skipped on failure: silently dropping tickers the
        # network hiccuped on would change the sample (and the shuffle order)
        # run to run. _get already retries transient errors; if it still
        # fails, stop and let the next run rebuild the full list.
        filings = edgar_client.list_10qs(ticker, MAX_10QS_PER_TICKER)
        if not filings:
            print(f"  {ticker}: no 10-Qs found on EDGAR")
        cases += [Case(ticker, f) for f in filings if FILED_FROM <= f.filing_date <= cutoff]
    cases.sort(key=lambda c: (c.filing.filing_date, c.ticker))   # stable base before the seeded shuffle
    random.Random(SHUFFLE_SEED).shuffle(cases)
    return cases


def _analyze(case: Case, model: str, computed: dict, filing_text: str, snapshot) -> pipeline.AnalysisRun:
    try:
        return pipeline.run_analysis(
            company=case.filing.company_name, ticker=case.ticker, report_date=case.filing.report_date,
            ratios=computed, filing_text=filing_text, market=snapshot, as_of_date=case.filing.filing_date,
        )
    except ClientError as exc:
        if getattr(exc, "status", None) == "RESOURCE_EXHAUSTED":
            raise QuotaExhausted(f"{model}: {str(exc)[:160]}") from exc
        raise
    except ServerError as exc:
        raise ModelUnavailable(f"{model}: {str(exc)[:160]}") from exc


def run_case(case: Case, model: str) -> tuple[BacktestRow, dict]:
    filing = case.filing
    raw_facts = edgar_client.company_facts(filing.cik)
    computed = ratios.compute_ratios(ratios.extract_facts(facts_as_of(raw_facts, filing.filing_date)))
    filing_text = edgar_client.fetch_filing_text(filing)
    with _yf_lock:
        snapshot = market_data.fetch_market_snapshot(case.ticker, as_of_date=filing.filing_date)
    ret = price_return(case.ticker, filing.filing_date, HORIZON_DAYS)
    spy = price_return(BENCHMARK, filing.filing_date, HORIZON_DAYS)
    if ret is None or spy is None:   # checked BEFORE spending LLM quota on the case
        raise RuntimeError(f"no {HORIZON_DAYS}-day return available (ticker={ret}, SPY={spy})")
    analysis = _analyze(case, model, computed, filing_text, snapshot)

    stance = parse_stance(analysis.recommendation)
    if stance == "Unparseable":   # found live: blank/garbled answers were being saved as results
        raise RuntimeError(f"no Bullish/Neutral/Bearish stance in the final step ({len(analysis.recommendation)} chars)")
    excess = ret - spy
    row = BacktestRow(
        ticker=case.ticker, company=filing.company_name,
        filing_date=filing.filing_date, report_date=filing.report_date, stance=stance,
        actual_return_pct=_pct(ret), spy_return_pct=_pct(spy), excess_return_pct=_pct(excess),
        hit=_directional_hit(stance, ret), hit_vs_market=_directional_hit(stance, excess),
        models=model,
    )
    log_entry = {
        "ticker": case.ticker, "filing_date": filing.filing_date, "report_date": filing.report_date,
        "models": model, "recommendation": analysis.recommendation,
        "steps": [{"title": s.title, "output": s.output} for s in analysis.steps],
    }
    return row, log_entry


# ── Persistence (upsert by (ticker, filing_date), written after every case) ──
_save_lock = threading.Lock()


def _key(record: dict) -> tuple[str, str]:
    return record["ticker"], record["filing_date"]


def load_done(models: list[str]) -> set[tuple[str, str]]:
    """Cases already finished by one of `models`: in results AND in the raw
    log. Rows from older harness versions (no model recorded) get rerun."""
    if not RESULTS_CSV.exists() or not RAW_LOG.exists():
        return set()
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        scored = {_key(r) for r in csv.DictReader(f) if r.get("models") in models}
    logged = {_key(e) for e in json.loads(RAW_LOG.read_text(encoding="utf-8"))}
    return scored & logged


def write_progress(total_in_window: int, models: list[str]) -> None:
    # every scored filing counts, whichever model answered it (not just this run's models)
    done = 0
    if RESULTS_CSV.exists():
        with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
            done = sum(1 for r in csv.DictReader(f) if r.get("models"))
    PROGRESS_JSON.write_text(json.dumps({
        "filings_in_window": total_in_window, "filings_done": done,
        "filed_from": FILED_FROM, "horizon_days": HORIZON_DAYS, "models": models,
        "tickers": len(TICKERS), "updated": datetime.now().isoformat(timespec="minutes"),
    }, indent=1), encoding="utf-8")


def save_case(row: BacktestRow, log_entry: dict) -> None:
    with _save_lock:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        existing_log = json.loads(RAW_LOG.read_text(encoding="utf-8")) if RAW_LOG.exists() else []
        merged_log = [e for e in existing_log if _key(e) != _key(log_entry)] + [log_entry]
        RAW_LOG.write_text(json.dumps(merged_log, indent=1), encoding="utf-8")

        rows: dict[tuple[str, str], dict] = {}
        if RESULTS_CSV.exists():
            with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
                rows = {_key(r): r for r in csv.DictReader(f)}
        rows = {**rows, _key(asdict(row)): asdict(row)}
        ordered = sorted(rows.values(), key=lambda r: (r["filing_date"], r["ticker"]))
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ordered)


# ── Workers ────────────────────────────────────────────────────────────────
_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def worker(model: str, work: queue.Queue, stats: dict[str, int]) -> None:
    strikes = 0
    with gemini_client.pinned_model(model):
        while True:
            try:
                case = work.get_nowait()
            except queue.Empty:
                return
            started = time.monotonic()
            try:
                row, entry = run_case(case, model)
            except QuotaExhausted as exc:
                work.put(case)
                log(f"[{model}] rate-limited after retries, retiring for this run ({exc})")
                return
            except ModelUnavailable as exc:
                work.put(case)
                strikes += 1
                if strikes >= MAX_UNAVAILABLE_STRIKES:
                    log(f"[{model}] still overloaded after {strikes} cooldowns, retiring ({exc})")
                    return
                log(f"[{model}] overloaded, cooling down {UNAVAILABLE_COOLDOWN_SECONDS}s")
                time.sleep(UNAVAILABLE_COOLDOWN_SECONDS)
                continue
            except Exception as exc:  # noqa: BLE001 -- a data problem with one filing; skip it
                log(f"[{model}] {case.ticker} {case.filing.filing_date} FAILED: {type(exc).__name__}: {exc}")
                continue
            strikes = 0
            save_case(row, entry)
            stats[model] = stats.get(model, 0) + 1
            log(f"[{model}] {row.ticker} {row.filing_date} -> {row.stance:8s} ret {row.actual_return_pct}% "
                f"vs SPY {row.excess_return_pct}% ({time.monotonic() - started:.0f}s, "
                f"~{work.qsize()} queued)")


def select_api_key(require_own_key: bool) -> bool:
    """Use the backtest's own key when set, so the batch run never spends the
    live app's free-tier quota. Returns False if the key is required and
    missing (the scheduled run passes --require-backtest-key)."""
    key = os.getenv(BACKTEST_KEY_ENV)
    if key:
        gemini_client.use_api_key(key)
        log(f"Using {BACKTEST_KEY_ENV} (separate from the live app's key).")
        return True
    if require_own_key:
        log(f"{BACKTEST_KEY_ENV} is not set -- refusing to spend the live app's quota. Add it to .env.")
        return False
    log(f"WARNING: {BACKTEST_KEY_ENV} not set; using the shared GEMINI_API_KEY.")
    return True


def run(tickers: list[str], limit: int | None, models: list[str]) -> None:
    gemini_client.SAME_MODEL_RETRY_WAITS = SAME_MODEL_RETRY_WAITS
    log(f"Models: {', '.join(models)}. Building worklist from EDGAR...")
    cases = build_worklist(tickers)
    done = load_done(models)
    remaining = [c for c in cases if (c.ticker, c.filing.filing_date) not in done]
    todo = remaining[:limit] if limit is not None else remaining
    log(f"{len(cases)} filings in window, {len(cases) - len(remaining)} done, running {len(todo)} now.")
    write_progress(len(cases), models)

    work: queue.Queue = queue.Queue()
    for case in todo:
        work.put(case)
    stats: dict[str, int] = {}
    threads = [threading.Thread(target=worker, args=(m, work, stats), name=m, daemon=True) for m in models]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"This run finished {sum(stats.values())} filings: {stats}. {work.qsize()} left for the next run.")
    write_progress(len(cases), models)
    summarize(models)


def summarize(models: list[str]) -> None:
    if not RESULTS_CSV.exists():
        return
    with RESULTS_CSV.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("models") in models]

    def rate(field: str) -> str:
        scored = [r[field] for r in rows if r.get(field) in ("True", "False")]
        hits = scored.count("True")
        return f"{hits}/{len(scored)} = {100 * hits / len(scored):.1f}%" if scored else "n/a"

    stances = {s: sum(r["stance"] == s for r in rows) for s in ("Bullish", "Neutral", "Bearish", "Unparseable")}
    print("\n" + "=" * 60)
    print(f"Filings scored in total: {len(rows)}  stances: {stances}")
    print(f"Directional hit rate (raw return):  {rate('hit')}")
    print(f"Directional hit rate (vs SPY):      {rate('hit_vs_market')}")
    print("=" * 60)


def acquire_run_lock():
    """One backtest at a time: a daily scheduled run and a manual run would
    otherwise score the same filings twice and race on the output files.
    Returns the open lock handle (keep it alive), or None if already held."""
    import msvcrt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(OUT_DIR / ".backtest.lock", "w")
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return None
    return handle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Only these tickers instead of the full fixed universe.")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Run at most N not-yet-done filings (smoke test with --limit 1).")
    parser.add_argument("--models", nargs="+", default=MODELS, help="Models to run (one worker each).")
    parser.add_argument("--require-backtest-key", action="store_true",
                        help=f"Exit unless {BACKTEST_KEY_ENV} is set (used by the scheduled run).")
    args = parser.parse_args()
    if not select_api_key(args.require_backtest_key):
        sys.exit(1)
    lock = acquire_run_lock()
    if lock is None:
        print("Another backtest run is already in progress -- exiting.")
        sys.exit(0)
    run(args.tickers or TICKERS, args.limit, args.models)
