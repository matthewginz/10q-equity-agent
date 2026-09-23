"""
Runs one or two ticker analyses IN PARALLEL with a live progress view.

Each ticker's fetch + 6-step pipeline runs on its own worker thread and only
records progress into a RunState -- Streamlit calls are not safe from
background threads, so the script thread polls those states and redraws:

  - a progress bar per ticker, with the current step and a time estimate
    (a typical-run guess until the first step lands, then a live estimate
    from the measured pace of the steps so far)
  - an aligned step grid: step N for every ticker sits in the same row, so
    two companies' analyses read side by side while they fill in

Before this, compare mode ran the two tickers one after the other and
stacked each one's step output in a single column (feedback from a real
user: "keep that information side by side, all steps aligned").
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import requests
import streamlit as st
from google.genai.errors import ClientError, ServerError

from core import edgar_client, market_data, ratios as ratios_mod
from services.pipeline import AnalysisRun, run_analysis

DATA_STAGES = [
    "Finding the latest 10-Q on SEC EDGAR",
    "Reading the filing's XBRL financials",
    "Reading MD&A and risk factors",
    "Getting the current stock price",
]
PIPELINE_STEPS = 6
DATA_SHARE = 0.10              # data fetches are quick; the 6 LLM steps are ~90% of the wait
TYPICAL_RUN_SECONDS = 90       # shown until the first LLM step finishes, then replaced by a live estimate
POLL_SECONDS = 0.5


def escape_markdown_math(text: str) -> str:
    """Found live: the LLM's own prose routinely writes two dollar amounts in
    one sentence ('$34.37 billion... $2.46 billion'), and Streamlit's
    markdown renders a paired '$...$' as inline LaTeX math, silently eating
    the text between them. Escaping every literal '$' turns off math mode
    without changing anything the reader sees."""
    return text.replace("$", "\\$")


@dataclass
class Bundle:
    ticker: str
    filing: object
    facts_json: dict
    computed: dict
    run: AnalysisRun
    market: "market_data.MarketSnapshot | None"


class UserFacingError(RuntimeError):
    """A failure with a message written for the person using the app."""


@dataclass
class RunState:
    """Progress for one ticker. Written only by its worker thread, read by the
    script thread; plain attribute writes and list appends are atomic under
    the GIL, and `version` tells the reader when the step grid changed."""
    ticker: str
    company: str = ""
    stage: int = 0
    started: float = field(default_factory=time.monotonic)
    pipeline_started: float | None = None
    last_step_at: float | None = None
    current_title: str = ""
    steps: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None
    bundle: Bundle | None = None
    done: bool = False
    finished_at: float | None = None
    version: int = 0

    def bump(self) -> None:
        self.version += 1


# ── Worker side (no Streamlit calls) ─────────────────────────────────────
def _fetch_filing(state: RunState):
    ticker = state.ticker
    try:
        filing = edgar_client.latest_10q(ticker)
    except requests.exceptions.RequestException as exc:
        raise UserFacingError(f"Couldn't reach SEC EDGAR for {ticker}: {exc}. It may be rate-limiting or "
                              "temporarily down — try again shortly.") from exc
    if filing is None:
        raise UserFacingError(
            f"No 10-Q found for '{ticker}' on SEC EDGAR. Check it's a US-listed filer that reports on Form 10-Q "
            "(foreign private issuers file 6-K/20-F instead, and won't resolve here)."
        )
    state.company = filing.company_name
    return filing


def _fetch_inputs(state: RunState):
    filing = _fetch_filing(state)
    state.stage = 1
    state.bump()
    try:
        facts_json = edgar_client.company_facts(filing.cik)
        computed = ratios_mod.compute_ratios(ratios_mod.extract_facts(facts_json))
        state.stage = 2
        state.bump()
        filing_text = edgar_client.fetch_filing_text(filing)
    except requests.exceptions.RequestException as exc:
        raise UserFacingError(f"Couldn't fetch {state.ticker}'s filing data from SEC EDGAR: {exc}. "
                              "Try again shortly.") from exc
    state.stage = 3
    state.bump()
    # Best effort: a market-data hiccup shouldn't block the analysis; the
    # pipeline degrades to fundamentals-only when the snapshot is None.
    try:
        snapshot = market_data.fetch_market_snapshot(state.ticker)
    except Exception:  # noqa: BLE001
        snapshot = None
    return filing, facts_json, computed, filing_text, snapshot


def _run_pipeline(state: RunState, filing, computed: dict, filing_text: str, snapshot) -> AnalysisRun:
    def on_step(i: int, title: str) -> None:
        state.current_title = title
        state.bump()

    def on_step_done(i: int, title: str, output: str) -> None:
        state.steps.append((title, output))
        state.last_step_at = time.monotonic()
        state.bump()

    state.pipeline_started = state.last_step_at = time.monotonic()
    try:
        return run_analysis(filing.company_name, state.ticker, filing.report_date, computed, filing_text,
                            market=snapshot, on_step=on_step, on_step_done=on_step_done)
    except (ClientError, ServerError) as exc:
        raise UserFacingError(
            f"The free-tier LLM pipeline failed on every fallback model for {state.ticker}: {exc}. "
            "Free-tier quota resets daily -- try again later, or try a different ticker."
        ) from exc


def _worker(state: RunState) -> None:
    try:
        filing, facts_json, computed, filing_text, snapshot = _fetch_inputs(state)
        run = _run_pipeline(state, filing, computed, filing_text, snapshot)
        state.bundle = Bundle(state.ticker, filing, facts_json, computed, run, snapshot)
    except UserFacingError as exc:
        state.error = str(exc)
    except Exception as exc:  # noqa: BLE001 -- surface anything else instead of hanging the progress view
        state.error = f"Unexpected error while analyzing {state.ticker}: {type(exc).__name__}: {exc}"
    finally:
        state.finished_at = time.monotonic()
        state.done = True
        state.bump()


# ── Script-thread side (all Streamlit calls) ─────────────────────────────
def _fmt_seconds(seconds: float) -> str:
    seconds = max(5, int(round(seconds / 5) * 5))
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60:02d}s"


def progress_of(state: RunState, now: float) -> tuple[float, str]:
    """(fraction complete, label) for one ticker's progress bar."""
    if state.error:
        return 1.0, "Stopped"
    if state.done:
        return 1.0, f"Done in {_fmt_seconds(state.finished_at - state.started)}"
    if state.pipeline_started is None:
        frac = DATA_SHARE * state.stage / len(DATA_STAGES)
        return frac, f"{DATA_STAGES[state.stage]}… (usually about {_fmt_seconds(TYPICAL_RUN_SECONDS)} total)"
    done = len(state.steps)
    frac = DATA_SHARE + (1 - DATA_SHARE) * done / PIPELINE_STEPS
    step_label = f"Step {min(done + 1, PIPELINE_STEPS)}/{PIPELINE_STEPS}: {state.current_title}"
    if done:
        pace = (state.last_step_at - state.pipeline_started) / done
        remaining = pace * (PIPELINE_STEPS - done) - (now - state.last_step_at)
    else:
        remaining = TYPICAL_RUN_SECONDS - (now - state.started)
    return frac, f"{step_label} · about {_fmt_seconds(remaining)} left"


def _render_bars(states: list[RunState]) -> None:
    now = time.monotonic()
    for col, state in zip(st.columns(len(states), gap="large"), states):
        frac, label = progress_of(state, now)
        with col:
            st.markdown(f"**{state.company or state.ticker}** ({state.ticker})")
            st.progress(min(frac, 1.0), text=label)
            if state.error:   # say what went wrong now, not only after the other side finishes
                st.error(state.error)


def _render_step_grid(states: list[RunState]) -> None:
    """Row N holds step N for every ticker, so the columns never drift apart."""
    reached = max(len(s.steps) + (0 if s.done else 1) for s in states)
    for i in range(min(reached, PIPELINE_STEPS)):
        cols = st.columns(len(states), gap="large")
        for col, state in zip(cols, states):
            with col:
                if i < len(state.steps):
                    title, output = state.steps[i]
                    with st.container(border=True):
                        st.markdown(f"**✓ {title}**")
                        st.markdown(escape_markdown_math(output))
                elif i == len(state.steps) and not state.done and state.pipeline_started is not None:
                    st.caption(f"⏳ Step {i + 1}: {state.current_title}…")


def run_with_progress(tickers: list[str]) -> list[Bundle | None]:
    """Analyze every ticker in parallel, showing live progress; returns one
    Bundle (or None on failure, with the error already shown) per ticker."""
    states = [RunState(t) for t in tickers]
    for state in states:
        threading.Thread(target=_worker, args=(state,), daemon=True).start()

    bars, grid = st.empty(), st.empty()
    drawn_versions: tuple[int, ...] | None = None
    while True:
        with bars.container():
            _render_bars(states)
        versions = tuple(s.version for s in states)
        if versions != drawn_versions:
            with grid.container():
                _render_step_grid(states)
            drawn_versions = versions
        if all(s.done for s in states):
            break
        time.sleep(POLL_SECONDS)

    bars.empty()
    grid.empty()
    for state in states:
        if state.error:
            st.error(state.error)
    return [s.bundle for s in states]
