"""
10-Q Equity Research Agent -- "Analyze" page (routed from app.py).

Enter any US-listed ticker or company name -> pulls its latest real 10-Q
from SEC EDGAR (free, public, no key) -> computes standardized financial
ratios from the filing's actual XBRL data -> runs a 6-step LLM analysis
pipeline (services/pipeline.py) -> renders a quantitative dashboard, risk
synthesis, and a final equity research stance. Supports comparing two
tickers side by side.

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io, pointed at this repo, app.py as
               the entrypoint, GEMINI_API_KEY set in the app's Secrets.
"""

import hashlib
import math

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from google.genai.errors import ClientError, ServerError
from streamlit_searchbox import st_searchbox

from core import edgar_client, ratios as ratios_mod
from services.pipeline import compare_stances
from ui.live_run import Bundle, escape_markdown_math, run_with_progress

# ── Styling ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
    --accent: #2563eb;
    --bull: #16a34a;
    --bear: #dc2626;
    --neutral: #6b7280;
}
/* layout="wide" above already tells Streamlit to use the full viewport,
   but this rule was still capping the actual content column at 1200px --
   fine on a laptop, wasteful on a real desktop monitor. This app is used
   mostly at a desk (unlike Kaiser-Stats, which is mostly phone), so widen
   the working area instead of defaulting to a narrow, scroll-heavy column. */
.block-container { padding-top: 2rem; max-width: 1600px; }
.stance-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
.stance-bullish { background: rgba(22,163,74,0.12); color: var(--bull); border: 1px solid rgba(22,163,74,0.35); }
.stance-bearish { background: rgba(220,38,38,0.12); color: var(--bear); border: 1px solid rgba(220,38,38,0.35); }
.stance-neutral { background: rgba(107,114,128,0.12); color: var(--neutral); border: 1px solid rgba(107,114,128,0.35); }
.company-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    border-radius: 0.75rem;
    background: rgba(37,99,235,0.06);
    border: 1px solid rgba(37,99,235,0.15);
    margin-bottom: 1rem;
}
.company-card-text b { font-size: 1.05rem; }
.gap-note {
    font-size: 0.85rem;
    color: var(--neutral);
    border-left: 3px solid var(--neutral);
    padding-left: 0.75rem;
    margin-top: 0.5rem;
}
.step-recap {
    border-left: 3px solid var(--accent);
    padding: 0.4rem 0 0.4rem 0.9rem;
    margin: 0.5rem 0 1rem 0;
}
.kpi-win { color: var(--bull); font-weight: 700; }
.market-card {
    padding: 1rem 1.25rem;
    border-radius: 0.75rem;
    background: rgba(37,99,235,0.04);
    border: 1px solid rgba(37,99,235,0.12);
    margin-bottom: 1rem;
}
.market-price { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.02em; }
.market-asof { font-size: 0.8rem; color: var(--neutral); font-weight: 500; margin-left: 0.5rem; }
.market-deltas { display: flex; gap: 1.25rem; margin: 0.3rem 0 0.85rem; font-size: 0.9rem; font-weight: 600; }
.delta-pos { color: var(--bull); }
.delta-neg { color: var(--bear); }
.range-track {
    position: relative;
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(220,38,38,0.30), rgba(107,114,128,0.25), rgba(22,163,74,0.30));
}
.range-marker {
    position: absolute;
    top: -4px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.35);
    transform: translateX(-50%);
}
.range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--neutral);
    margin-top: 0.45rem;
}
.ratio-table { width: 100%; border-collapse: collapse; margin-bottom: 0.85rem; }
.ratio-table td { padding: 0.35rem 0.6rem; border-bottom: 1px solid rgba(107,114,128,0.15); font-size: 0.9rem; }
.ratio-label { color: var(--neutral); text-transform: capitalize; }
.ratio-value { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }
.cell-pos { background: rgba(22,163,74,0.10); color: var(--bull); border-radius: 0.3rem; }
.cell-neg { background: rgba(220,38,38,0.10); color: var(--bear); border-radius: 0.3rem; }
.verdict-card {
    background: rgba(37,99,235,0.05);
    border: 1px solid rgba(37,99,235,0.2);
    border-left: 4px solid var(--accent);
    border-radius: 0.5rem;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 1.5rem;
}
h1, h2, h3 { letter-spacing: -0.01em; }
</style>
""",
    unsafe_allow_html=True,
)

# Compare mode's company cards + expand/collapse controls stay pinned under
# Streamlit's 3.75rem top bar while scrolling. Found live: sticky on the keyed
# container itself does nothing -- Streamlit wraps it in an stLayoutWrapper of
# exactly its height -- so the wrapper is what gets pinned.
_IS_DARK = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
_PAGE_BG = "#0e1117" if _IS_DARK else "#ffffff"
_INK = "#fafafa" if _IS_DARK else "#31333f"
st.markdown(
    f"""
<style>
div[data-testid="stLayoutWrapper"]:has(> .st-key-compare-header) {{ position: sticky; top: 3.75rem; z-index: 99; }}
.st-key-compare-header {{ background: {_PAGE_BG}; padding: 0.5rem 0 0.25rem;
    border-bottom: 1px solid rgba(128,128,128,0.25); }}
.st-key-compare-header .company-card {{ margin-bottom: 0; }}
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ───────────────────────────────────────────────────────────────
st.title("\U0001F4C8 10-Q Equity Research Agent")

with st.expander("How this actually works", expanded=False):
    st.markdown(
        "1. **SEC EDGAR** — resolves the ticker to a CIK, pulls the latest real 10-Q and its structured XBRL "
        "financial facts (the same standardized data every US public company reports).\n"
        "2. **Ratio engine** — computes profitability, liquidity, leverage, and cash-flow ratios directly from "
        "that XBRL data, with duration- and period-alignment checks so a quarterly figure never gets silently "
        "mixed with an annual one.\n"
        "3. **6-step LLM pipeline** — quantitative snapshot → risk/MD&A synthesis → segment & forward-looking "
        "detail → narrative-vs-numbers consistency check → capital allocation read → final equity stance "
        "(sentiment, price reconciliation, and a recommendation, grounded in real current market data). Each "
        "step is grounded in the real prior steps, not six independent prompts against the same raw dump, and "
        "each step's finding is shown live as it completes."
    )

# ── Company logos (curated domains for Clearbit's free logo API; anything
#    not in this map falls back to a generated initials badge, never a
#    broken image) ─────────────────────────────────────────────────────────
_LOGO_DOMAINS = {
    "AAPL": "apple.com", "MSFT": "microsoft.com", "GOOGL": "abc.xyz", "GOOG": "abc.xyz",
    "AMZN": "amazon.com", "META": "meta.com", "NVDA": "nvidia.com", "TSLA": "tesla.com",
    "JPM": "jpmorganchase.com", "BAC": "bankofamerica.com", "WFC": "wellsfargo.com",
    "GS": "goldmansachs.com", "MS": "morganstanley.com", "C": "citigroup.com",
    "KO": "coca-colacompany.com", "PEP": "pepsico.com", "MCD": "mcdonalds.com",
    "SBUX": "starbucks.com", "NKE": "nike.com", "DIS": "disney.com",
    "NFLX": "netflix.com", "V": "visa.com", "MA": "mastercard.com",
    "PYPL": "paypal.com", "ADBE": "adobe.com", "CRM": "salesforce.com",
    "ORCL": "oracle.com", "INTC": "intel.com", "AMD": "amd.com",
    "IBM": "ibm.com", "CSCO": "cisco.com", "QCOM": "qualcomm.com",
    "T": "att.com", "VZ": "verizon.com", "XOM": "exxonmobil.com",
    "CVX": "chevron.com", "WMT": "walmart.com", "TGT": "target.com",
    "HD": "homedepot.com", "COST": "costco.com", "UNH": "unitedhealthgroup.com",
    "JNJ": "jnj.com", "PFE": "pfizer.com", "MRK": "merck.com",
    "BA": "boeing.com", "GE": "ge.com", "F": "ford.com", "GM": "gm.com",
    "UBER": "uber.com", "ABNB": "airbnb.com", "SHOP": "shopify.com",
    "SQ": "squareup.com", "COIN": "coinbase.com", "SPOT": "spotify.com",
    "SNAP": "snap.com", "PINS": "pinterest.com", "RIVN": "rivian.com",
    "BRK-B": "berkshirehathaway.com",
}


def _avatar_color(ticker: str) -> str:
    """Deterministic color from the ticker so the same symbol always gets
    the same fallback badge color across runs (not random per render)."""
    digest = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
    hue = digest % 360
    return f"hsl({hue}, 62%, 46%)"


def _avatar_badge(ticker: str, size: int) -> str:
    initials = ticker[:2]
    color = _avatar_color(ticker)
    return (
        f'<div style="display:flex;width:{size}px;height:{size}px;border-radius:14px;'
        f"background:{color};color:white;align-items:center;justify-content:center;"
        f'font-weight:800;font-size:{size * 0.34}px;letter-spacing:-0.02em;">{initials}</div>'
    )


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _logo_url(domain: str) -> str | None:
    """Google's favicon service (no key, no signup) -- found live: Clearbit's
    free logo API is dead (logo.clearbit.com doesn't even resolve in DNS
    anymore, discontinued after the 2023 HubSpot acquisition), and Streamlit's
    markdown sanitizer strips inline onerror handlers, so a client-side <img
    onerror> fallback can't paper over a dead source either way. Checked
    server-side once per domain per day; only returns a URL confirmed to
    actually resolve, so a broken image never reaches the page."""
    url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
    try:
        r = requests.get(url, timeout=3, stream=True)
        ok = r.status_code == 200 and (r.headers.get("content-type") or "").startswith("image")
        r.close()
        return url if ok else None
    except requests.exceptions.RequestException:
        return None


def company_avatar_html(ticker: str, size: int = 56) -> str:
    """A real company favicon (Google's favicon service, keyed off a curated
    domain map), pre-checked server-side so an unresolvable logo never
    reaches the page as a broken image -- falls back to a generated
    initials badge instead."""
    ticker = ticker.upper()
    domain = _LOGO_DOMAINS.get(ticker)
    url = _logo_url(domain) if domain else None
    if url:
        img = (
            f'<img src="{url}" '
            f'style="width:{size}px;height:{size}px;border-radius:14px;object-fit:contain;'
            f'background:white;padding:8px;box-sizing:border-box;" />'
        )
        return f'<div style="display:inline-block;">{img}</div>'
    return _avatar_badge(ticker, size)


# ── Ticker search (one real combobox, ranked on the ticker itself) ────────
#
# Streamlit's own selectbox filters its options client-side by unranked
# substring match against whatever text is displayed, so typing "BE" could
# surface "ABEO" (contains "be") ahead of the actual exact ticker "BE" --
# and there's no way to fix that from inside a plain selectbox, since its
# filtering runs in the browser, not in this script. A separate text_input
# next to it fixed the ranking but made it two boxes instead of one search
# experience. streamlit_searchbox is a single widget that calls a real
# Python function on every keystroke and renders ITS return value as the
# dropdown, so the same box both takes the input and shows the live-ranked
# results -- and the ranking below only ever looks at the ticker itself:
# exact ticker > ticker starts with the query > ticker contains the query.
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def _load_ticker_directory() -> list[tuple[str, str]]:
    directory = edgar_client.company_directory()
    seen: dict[str, str] = {}
    for t, name, _cik in directory:
        if t not in seen:
            seen[t] = name
    return sorted(seen.items())


def _ticker_rank(query: str, ticker: str) -> int:
    """Lower is a better match; 3 means "no match" and gets dropped."""
    if ticker == query:
        return 0
    if ticker.startswith(query):
        return 1
    if query in ticker:
        return 2
    return 3


def _search_tickers(query: str, directory: list[tuple[str, str]], limit: int = 20) -> list[tuple[str, str]]:
    """Returns (display_label, ticker) pairs -- st_searchbox shows the label
    in the dropdown and returns the matching ticker once one is picked."""
    q = query.strip().upper()
    if not q:
        return [(f"{t} — {name}", t) for t, name in directory[:limit]]
    ranked = sorted(
        ((_ticker_rank(q, t), t, name) for t, name in directory if _ticker_rank(q, t) < 3),
        key=lambda r: (r[0], r[1]),
    )
    return [(f"{t} — {name}", t) for _, t, name in ranked[:limit]]


TICKER_DIRECTORY = _load_ticker_directory()

compare_mode = st.checkbox("Compare with a second ticker")

col1, col2 = st.columns(2) if compare_mode else (st.container(), None)
with col1:
    ticker1 = st_searchbox(
        lambda q: _search_tickers(q, TICKER_DIRECTORY),
        placeholder="Type a ticker (e.g. AAPL, BE, MSFT)...",
        label="Ticker" if not compare_mode else "First ticker",
        key="ticker_searchbox_1",
        clear_on_submit=False,
    ) or ""
ticker1 = ticker1.strip().upper()

ticker2 = ""
if compare_mode:
    with col2:
        ticker2 = st_searchbox(
            lambda q: _search_tickers(q, TICKER_DIRECTORY),
            placeholder="Type a second ticker...",
            label="Second ticker",
            key="ticker_searchbox_2",
            clear_on_submit=False,
        ) or ""
    ticker2 = ticker2.strip().upper()

run_clicked = st.button(
    "Run analysis" if not compare_mode else "Compare",
    type="primary",
    disabled=not ticker1 or (compare_mode and not ticker2),
)

STANCE_CLASS = {"BULLISH": "stance-bullish", "BEARISH": "stance-bearish"}


def stance_badge(text: str) -> str:
    # Found live: scanning the FULL multi-paragraph recommendation let a
    # later hedging phrase ("...rather than a fully bullish read") override
    # the actually-stated stance -- one real case showed a BULLISH badge on
    # a recommendation whose own heading said "Final Equity Research
    # Stance: Neutral". The model always states its stance up front (see
    # the step-5 prompt in services/pipeline.py), so only search that
    # opening window, not the whole text.
    opening = text[:150].upper()
    cls = next((v for k, v in STANCE_CLASS.items() if k in opening), "stance-neutral")
    label = next((k for k in STANCE_CLASS if k in opening), "NEUTRAL")
    return f'<span class="stance-badge {cls}">{label}</span>'


def fmt_ratio(key: str, value) -> str:
    """Found live: XBRL values come back as plain Python ints whenever the
    filer's own value had no decimal point (most raw dollar figures) -- an
    `isinstance(value, float)` check silently skipped all of those and fell
    through to a raw, unformatted number (e.g. free cash flow rendered as
    literal `5570000000` instead of `$5.57B`). Checking (int, float)
    together fixes every dollar-amount metric, not just the ones that
    happened to come back as floats."""
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        if any(w in key for w in ("margin", "yoy", "ratio")) and abs(value) < 3:
            return f"{value:+.1%}" if "yoy" in key else f"{value:.1%}"
        if abs(value) >= 1_000_000:
            return f"${value / 1e9:,.2f}B"
        return f"{value:.2f}"
    return str(value)


# ── Render building blocks -- each renders into whatever container is
#    currently active (the caller opens `with column:` where it matters),
#    shared verbatim by single-ticker mode and compare mode. ────────────────
def render_company_card(bundle: Bundle) -> None:
    filing = bundle.filing
    st.markdown(
        f'<div class="company-card">{company_avatar_html(bundle.ticker)}'
        f'<div class="company-card-text"><b>{filing.company_name}</b> ({bundle.ticker})<br/>'
        f"10-Q for period ending {filing.report_date}, filed {filing.filing_date}. "
        f'<a href="{filing.document_url}" target="_blank">View the real filing on SEC.gov →</a></div></div>',
        unsafe_allow_html=True,
    )


def render_market_snapshot(bundle: Bundle) -> None:
    """The price/valuation context the pipeline's final stance step actually
    reasons from (services/pipeline.py step 6) -- fetched in ui/live_run.py
    but, until this function existed, never shown anywhere in the UI. Best
    effort: renders nothing if the market fetch failed, matching the same
    "None means not available" contract as the rest of this app rather than
    showing a broken or fabricated card."""
    m = bundle.market
    if m is None or not math.isfinite(m.price):   # never render "$nan"
        return
    span = m.fifty_two_week_high - m.fifty_two_week_low
    position_pct = max(0.0, min(100.0, (m.price - m.fifty_two_week_low) / span * 100)) if span > 0 else 50.0

    def _delta(label: str, pct: float | None) -> str:
        if pct is None:
            return ""
        cls = "delta-pos" if pct >= 0 else "delta-neg"
        return f'<span class="{cls}">{label}: {pct:+.1%}</span>'

    deltas = " ".join(d for d in (_delta("90d", m.return_90d_pct), _delta("365d", m.return_365d_pct)) if d)
    st.markdown(
        f'<div class="market-card">'
        f'<span class="market-price">${m.price:,.2f}</span>'
        f'<span class="market-asof">as of {m.as_of}</span>'
        f'<div class="market-deltas">{deltas}</div>'
        f'<div class="range-track"><div class="range-marker" style="left:{position_pct:.1f}%;"></div></div>'
        f'<div class="range-labels"><span>${m.fifty_two_week_low:,.2f}</span>'
        f'<span>52-week range</span><span>${m.fifty_two_week_high:,.2f}</span></div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_kpis(bundle: Bundle) -> None:
    prof, cf, growth = bundle.computed["profitability"], bundle.computed["cash_flow"], bundle.computed["yoy_growth"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue YoY", fmt_ratio("yoy", growth.get("revenue_yoy")) if growth.get("revenue_yoy") is not None else "n/a")
    k2.metric("Gross Margin", fmt_ratio("margin", prof.get("gross_margin")))
    k3.metric("Operating Margin", fmt_ratio("margin", prof.get("operating_margin")))
    k4.metric("Free Cash Flow", fmt_ratio("fcf", cf.get("free_cash_flow")))


def render_charts(bundle: Bundle) -> None:
    hist_col1, hist_col2 = st.columns(2)
    try:
        rev_hist = ratios_mod.extract_history(bundle.facts_json, "revenue", n_quarters=8)
        ni_hist = ratios_mod.extract_history(bundle.facts_json, "net_income", n_quarters=8)
        if rev_hist:
            with hist_col1:
                st.caption("Revenue, last 8 quarters ($)")
                st.line_chart({e: v for e, v in rev_hist})
        if ni_hist:
            with hist_col2:
                st.caption("Net income, last 8 quarters ($)")
                st.line_chart({e: v for e, v in ni_hist})
    except Exception:
        pass  # trend charts are a bonus visualization; never block the core analysis on them


def _ratio_cell_class(key: str, value) -> str:
    """Color-code only genuinely direction-unambiguous metrics (margins,
    YoY growth) green/red by sign -- deliberately NOT leverage/liquidity
    ratios, where "good" depends on the business (e.g. banks run high
    leverage by design; a red tint on that would be actively misleading,
    not helpful -- the reasoning-quality audit on this pipeline flagged
    exactly this as a real gap, so the UI shouldn't repeat it)."""
    if not isinstance(value, (int, float)):
        return ""
    if "margin" in key or "yoy" in key:
        return "cell-pos" if value >= 0 else "cell-neg"
    return ""


def render_ratio_breakdown(bundle: Bundle) -> None:
    with st.expander(f"{bundle.ticker} — full computed ratio breakdown", expanded=False):
        for section, values in bundle.computed.items():
            if section == "data_gaps":
                continue
            st.markdown(f"**{section.replace('_', ' ').title()}**")
            rows = "".join(
                f'<tr><td class="ratio-label">{k.replace("_", " ")}</td>'
                f'<td class="ratio-value {_ratio_cell_class(k, v)}">{fmt_ratio(k, v)}</td></tr>'
                for k, v in values.items()
            )
            st.markdown(f'<table class="ratio-table">{rows}</table>', unsafe_allow_html=True)
        if bundle.computed.get("data_gaps"):
            st.markdown(
                f'<div class="gap-note">Not tagged in this filing\'s XBRL data (reported as unavailable, '
                f'never estimated): {", ".join(bundle.computed["data_gaps"])}</div>',
                unsafe_allow_html=True,
            )


def render_stance(bundle: Bundle) -> None:
    st.subheader("Equity Research Stance")
    st.markdown(stance_badge(bundle.run.recommendation), unsafe_allow_html=True)
    st.markdown(escape_markdown_math(bundle.run.recommendation))


def render_steps(bundle: Bundle) -> None:
    st.subheader("Step-by-Step Analysis")
    tabs = st.tabs([s.title for s in bundle.run.steps])
    for tab, step in zip(tabs, bundle.run.steps):
        with tab:
            st.markdown(escape_markdown_math(step.output))


def render_single(bundle: Bundle) -> None:
    render_company_card(bundle)
    render_market_snapshot(bundle)
    render_kpis(bundle)
    render_charts(bundle)
    render_ratio_breakdown(bundle)
    st.divider()
    render_stance(bundle)
    render_steps(bundle)


# ── Compare-mode-only render blocks: these actually compare, not just
#    lay two independent analyses side by side. ─────────────────────────────
_KPI_COMPARE_DEFS = [
    ("Revenue YoY", "yoy", lambda c: c["yoy_growth"].get("revenue_yoy")),
    ("Gross Margin", "margin", lambda c: c["profitability"].get("gross_margin")),
    ("Operating Margin", "margin", lambda c: c["profitability"].get("operating_margin")),
    ("Free Cash Flow", "fcf", lambda c: c["cash_flow"].get("free_cash_flow")),
]


def render_kpi_comparison(b1: Bundle, b2: Bundle) -> None:
    # st.container(border=True), not a hand-rolled <div>, because a raw
    # unsafe_allow_html div opened in one st.markdown call and closed in a
    # later one does NOT wrap the native st.columns rows in between --
    # Streamlit mounts each st.markdown call as its own sibling DOM node,
    # so the browser just auto-closes the dangling tag. Found live.
    with st.container(border=True):
        header = st.columns([2, 1, 1])
        header[1].markdown(f"**{b1.ticker}**")
        header[2].markdown(f"**{b2.ticker}**")
        for label, fmt_key, getter in _KPI_COMPARE_DEFS:
            v1, v2 = getter(b1.computed), getter(b2.computed)
            # Only highlight a winner when BOTH sides have a real value to
            # compare -- found live: with the old (v2 is None or v1 > v2)
            # form, a lone value on one side (the other reported "n/a")
            # always highlighted green, even a negative YoY figure, which
            # reads as "this side won" when really there was nothing to
            # compare it against.
            both_present = v1 is not None and v2 is not None
            better1 = both_present and v1 > v2
            better2 = both_present and v2 > v1
            s1 = fmt_ratio(fmt_key, v1) if v1 is not None else "n/a"
            s2 = fmt_ratio(fmt_key, v2) if v2 is not None else "n/a"
            row = st.columns([2, 1, 1])
            row[0].markdown(label)
            row[1].markdown(f'<span class="{"kpi-win" if better1 else ""}">{s1}</span>', unsafe_allow_html=True)
            row[2].markdown(f'<span class="{"kpi-win" if better2 else ""}">{s2}</span>', unsafe_allow_html=True)


def render_chart_comparison(b1: Bundle, b2: Bundle) -> None:
    try:
        rev1 = dict(ratios_mod.extract_history(b1.facts_json, "revenue", n_quarters=8))
        rev2 = dict(ratios_mod.extract_history(b2.facts_json, "revenue", n_quarters=8))
        ni1 = dict(ratios_mod.extract_history(b1.facts_json, "net_income", n_quarters=8))
        ni2 = dict(ratios_mod.extract_history(b2.facts_json, "net_income", n_quarters=8))
    except Exception:
        return  # trend charts are a bonus visualization; never block the core comparison on them

    c1, c2 = st.columns(2)
    if rev1 or rev2:
        with c1:
            st.caption("Revenue, last 8 quarters ($) — overlaid")
            st.line_chart(pd.DataFrame({b1.ticker: rev1, b2.ticker: rev2}))
    if ni1 or ni2:
        with c2:
            st.caption("Net income, last 8 quarters ($) — overlaid")
            st.line_chart(pd.DataFrame({b1.ticker: ni1, b2.ticker: ni2}))


def render_verdict(b1: Bundle, b2: Bundle) -> None:
    st.subheader("Head-to-Head: Which Would You Rather Own?")
    try:
        with st.spinner("Weighing both theses against each other..."):
            verdict = compare_stances(
                b1.filing.company_name, b1.ticker, b1.run,
                b2.filing.company_name, b2.ticker, b2.run,
            )
    except RuntimeError as exc:
        st.error(str(exc))
        return
    except (ClientError, ServerError) as exc:
        st.error(f"Couldn't generate a head-to-head verdict: {exc}")
        return
    # Blank lines right after the opening tag matter: found live, a
    # markdown heading (###) glued directly to <div> with no blank line
    # doesn't get parsed as a block-level heading and renders as literal
    # text, even though a table a few lines later parses fine.
    st.markdown(f'<div class="verdict-card">\n\n{escape_markdown_math(verdict)}\n\n</div>', unsafe_allow_html=True)


def render_compare(b1: "Bundle | None", b2: "Bundle | None") -> None:
    if b1 is None or b2 is None:
        return  # a fetch/pipeline error for one side was already shown by run_with_progress
    render_compare_header(b1, b2)
    left, right = st.columns(2, gap="large")
    with left:
        render_market_snapshot(b1)
    with right:
        render_market_snapshot(b2)

    st.subheader("Key Metrics — Head to Head")
    render_kpi_comparison(b1, b2)
    render_chart_comparison(b1, b2)

    left, right = st.columns(2, gap="large")
    with left:
        render_ratio_breakdown(b1)
    with right:
        render_ratio_breakdown(b2)

    st.divider()
    render_verdict(b1, b2)

    st.divider()
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        render_stance(b1)
    with right, st.container(border=True):
        render_stance(b2)
    render_steps_side_by_side(b1, b2)


def render_steps_side_by_side(b1: Bundle, b2: Bundle) -> None:
    """One row per pipeline step with both companies in it, so step N of
    one always lines up with step N of the other (feedback from a real user:
    the old layout put each company's steps in its own column of tabs, which
    made comparing the same step across the two awkward)."""
    st.subheader("Step-by-Step, Side by Side")
    st.caption("Click a step to open or close it, or use Expand all / Collapse all in the header above.")
    with st.container(key=_STEPS_KEY):
        for s1, s2 in zip(b1.run.steps, b2.run.steps):
            with st.expander(s1.title, expanded=False):
                left, right = st.columns(2, gap="large")
                with left:
                    st.markdown(f"**{b1.ticker}**")
                    st.markdown(escape_markdown_math(s1.output))
                with right:
                    st.markdown(f"**{b2.ticker}**")
                    st.markdown(escape_markdown_math(s2.output))


_STEPS_KEY = "steps-side-by-side"


def render_compare_header(b1: Bundle, b2: Bundle) -> None:
    """Both company cards, pinned to the top while scrolling (CSS above),
    with expand/collapse-all for the side-by-side steps."""
    with st.container(key="compare-header"):
        left, right = st.columns(2, gap="large")
        with left:
            render_company_card(b1)
        with right:
            render_company_card(b2)
        _expand_collapse_controls()


def _expand_collapse_controls() -> None:
    """Runs in the browser, not as st.button: results only exist on the run
    that followed the Compare click, so a Streamlit rerun would wipe them.
    Clicks each step's <summary> (rather than setting `open`) so the
    expanders' own state stays in sync; a single step still toggles on its own."""
    button = (f"font:600 0.85rem system-ui,sans-serif;color:{_INK};background:transparent;cursor:pointer;"
              "border:1px solid rgba(128,128,128,0.45);border-radius:0.5rem;padding:0.3rem 0.8rem;")
    components.html(
        f"""
<div style="display:flex;gap:0.5rem;justify-content:flex-end;margin:0">
  <button style="{button}" onclick="setAll(true)">Expand all steps</button>
  <button style="{button}" onclick="setAll(false)">Collapse all steps</button>
</div>
<script>
function setAll(open) {{
  window.parent.document.querySelectorAll('.st-key-{_STEPS_KEY} details').forEach(d => {{
    if (d.open !== open) {{ const s = d.querySelector('summary'); if (s) s.click(); else d.open = open; }}
  }});
}}
</script>""",
        height=40,
    )


if run_clicked:
    if compare_mode:
        b1, b2 = run_with_progress([ticker1, ticker2])   # both tickers run in parallel
        if b1 is not None and b2 is not None:
            render_compare(b1, b2)
        elif b1 is not None or b2 is not None:
            # one side failed (its error is already shown) -- don't throw away the other
            survivor = b1 if b1 is not None else b2
            st.info(f"Showing {survivor.ticker} on its own, since the other analysis didn't finish.")
            render_single(survivor)
    else:
        (bundle,) = run_with_progress([ticker1])
        if bundle is not None:
            render_single(bundle)
