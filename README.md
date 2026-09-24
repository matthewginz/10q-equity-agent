# 10-Q Equity Research Agent

**Live app:** https://10q-equity-agent-esldzub5xgfc3j8mcc8u42.streamlit.app/

Enter any US-listed ticker. Pulls its real, latest 10-Q from SEC EDGAR
(free, public, no key), computes standard financial ratios from the
filing's actual structured (XBRL) data, then runs a 5-step LLM research
pipeline against the real filing text to produce a quantitative snapshot,
a risk synthesis, a narrative-vs-numbers consistency check, a capital
allocation read, and a final equity research stance.

## Why this exists

An earlier version of this ([`amazon-10q-agent`](https://github.com)) worked
against one hardcoded company's static PDF and Excel export. This version
works against any real US filer, live, using the same standardized data
every public company reports to the SEC — no manual data prep per company.

## Architecture

```
ticker input
    │
    ▼
core/edgar_client.py    ── SEC EDGAR: ticker → CIK → latest 10-Q filing
    │                       + structured XBRL financial facts
    ▼
core/ratios.py          ── profitability / liquidity / leverage / cash flow /
    │                       YoY ratios, computed from the filing's own XBRL
    │                       data (not hardcoded row labels) -- works for any
    │                       filer's actual reported figures
    ▼
services/pipeline.py    ── 5 sequential LLM steps, each grounded in the
    │                       previous steps' real output, not 5 independent
    │                       prompts against the same raw context
    ▼
app.py                  ── Streamlit router: views/analyze.py (live analysis)
                           + views/track_record.py (backtest results)
```

## Track record

The app's **Track Record** page scores the agent's calls at scale. The universe
is the S&P 500, frozen to `docs/backtest/universe.csv` before any results
existed. Every 10-Q those companies filed from July 2024 onward is a test case
(~3,000 filings, and the count grows as new filings reach their 90-day
horizon). Filings run newest quarter first. Recent quarters are the cleanest
test, because the model is least likely to have seen their outcomes in
training. A nightly scheduled run scores as many filings as the free quota
allows and pushes the results, so the live page updates on its own. For each
filing:

- the pipeline sees only the XBRL facts and price available on the filing
  date, and is told that date is "today"
- one Gemini model answers every step of that filing. Five free-tier flash
  models share the work, and each row records which model answered
- the call is scored against the stock's 90-day return, both raw and minus
  the S&P 500 (SPY), and against Wall Street's consensus on the same date,
  rebuilt from dated analyst ratings (`core/analyst_consensus.py`)

Every hit rate is shown with a 95% Wilson interval and an exact binomial test
against a coin flip (`core/stats.py`). There are also breakdowns by quarter
and by model, to catch an edge that only shows up in one of them.

Gemini's free tier caps each model at 20 requests/day (6 per filing), so the
run is resumable: filings go newest quarter first, results save after
every filing, and a daily scheduled task (`scripts/run_backtest_daily.bat`)
picks up where the last run stopped.

```bash
pip install -r requirements-backtest.txt
python scripts/backtest_accuracy.py --limit 1   # smoke test one filing
python scripts/backtest_accuracy.py             # as many as today's quota allows (resumable)
python scripts/analyst_comparison.py            # no LLM -> docs/backtest/analyst_comparison.csv
python -m pytest tests
```

### A few things worth knowing about how this actually works

- **Ratios are computed from real XBRL facts, not text-parsed from a PDF.**
  Every US public company tags its financial statements with standardized
  concepts (`us-gaap:Revenues`, `us-gaap:OperatingIncomeLoss`, etc.) as part
  of its SEC filing. `core/ratios.py` reads those directly.
- **XBRL duration handling is non-trivial and got it wrong on the first
  pass.** The same concept is tagged at multiple durations (a quarter, a
  6-month YTD, a full fiscal year) under one tag, and companies sometimes
  retire one tag name for another over time. An early version of this
  mixed a quarterly revenue figure against an annual cost figure and
  produced a nonsense 13% gross margin for Apple. Fixed by merging all
  fallback tag names before picking the freshest point (instead of
  stopping at the first tag with *any* data) and isolating a true single
  quarter — directly if tagged, or by subtracting the prior YTD figure
  from the current one when the filer only tags cash-flow items
  cumulatively (the common case).
- **Cross-metric ratios are period-aligned before they're computed.** If
  two facts feeding one ratio (e.g. operating income and interest expense)
  come from different reporting periods — which happens when a company
  stops tagging one line item — the ratio is reported as unavailable
  rather than silently computed from mismatched periods.
- **Bank filers correctly return `null` for ratios that don't apply to
  them.** JPMorgan doesn't report "cost of revenue" or "current assets"
  the way an industrial company does; rather than force industrial-company
  ratio formulas onto a bank's balance sheet, missing concepts are
  reported as gaps, not guessed at.
- **The filing-text fetch is anchored on the actual MD&A section heading,
  not a fixed byte offset.** A modern 10-Q's inline-XBRL markup can carry
  tens of thousands of characters of tagging metadata before any
  human-readable prose starts; slicing raw HTML by a fixed offset returned
  nothing but tag noise on the first pass. Fixed by stripping to plain
  text and searching for the real "Management's Discussion and Analysis"
  heading (the *last* occurrence, since the first is the table of
  contents).

## Run locally

```bash
python -m venv venv && venv\Scripts\activate      # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                               # add a free key from https://aistudio.google.com/apikey
streamlit run app.py
```

## Stack

Python · Streamlit · SEC EDGAR (free, public, no key) · Gemini API (free
tier, model-fallback chain across quota/availability) · BeautifulSoup

## License

MIT — see [LICENSE](LICENSE).
