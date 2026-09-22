"""
Track Record page -- how the agent's backtested stances held up, against
both what the stock actually did and what Wall Street was saying on the
same date.

Reads only committed files (no network, no LLM):
  docs/backtest/analyst_comparison.csv  <- scripts/analyst_comparison.py
  docs/backtest/raw_runs.json           <- scripts/backtest_accuracy.py

Chart color encodes stance everywhere on this page, with one validated
blue/gray/red set (Bullish/Neutral/Bearish). Deliberately NOT the green/red
used for badges on the Analyze page: green vs. red measured Delta E 5.0
under deuteranopia (below the 6 floor), blue vs. red passes in light and
dark mode.
"""

import json
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BACKTEST_DIR = Path(__file__).resolve().parent.parent / "docs" / "backtest"
STANCES = ["Bullish", "Neutral", "Bearish"]
STANCE_COLORS = ["#2a78d6", "#8b8b87", "#dc2626"]
STANCE_SCALE = alt.Scale(domain=STANCES, range=STANCE_COLORS)
STANCE_COLOR_MAP = dict(zip(STANCES, STANCE_COLORS))
NO_DATA = "No data"
NO_DATA_COLOR = "#d4d4d0"
# Chart text follows the active theme's ink, never a series color.
_IS_DARK = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
TEXT_COLOR = "#e5e7eb" if _IS_DARK else "#1f2937"
CELL_GAP_COLOR = "#0e1117" if _IS_DARK else "#ffffff"

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 1400px; }
.tr-lede { font-size: 1.02rem; opacity: 0.8; max-width: 60rem; margin: -0.5rem 0 1.5rem; }
.tr-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.9rem; margin-bottom: 1.75rem; }
.tr-kpi {
    padding: 1rem 1.15rem; border-radius: 0.75rem;
    border: 1px solid rgba(128,128,128,0.22); background: rgba(128,128,128,0.05);
}
.tr-kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.65; font-weight: 600; }
.tr-kpi-value { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
.tr-kpi-sub { font-size: 0.84rem; opacity: 0.7; margin-top: 0.15rem; }
.tr-callouts { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 0.9rem; margin: 0.25rem 0 0.5rem; }
.tr-callout { padding: 1rem 1.2rem; border-radius: 0.75rem; border: 1px solid rgba(128,128,128,0.22); border-left: 4px solid var(--tr-edge); }
.tr-callout h4 { margin: 0 0 0.35rem; font-size: 1rem; padding: 0; }
.tr-callout .tr-big { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.tr-callout p { margin: 0.35rem 0 0; font-size: 0.9rem; opacity: 0.85; }
.tr-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.6rem; }
.tr-chip {
    display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.8rem; font-weight: 600;
    padding: 0.15rem 0.55rem; border-radius: 999px; border: 1px solid rgba(128,128,128,0.25);
    font-variant-numeric: tabular-nums;
}
.tr-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.tr-caveat { font-size: 0.85rem; opacity: 0.75; border-left: 3px solid rgba(128,128,128,0.5); padding-left: 0.75rem; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    frame = pd.read_csv(BACKTEST_DIR / "analyst_comparison.csv")
    frame["street_label"] = frame["street_stance"].replace({"Insufficient": NO_DATA})
    frame["street_mix"] = frame.apply(
        lambda r: f"{r.n_buy} Buy · {r.n_hold} Hold · {r.n_sell} Sell" if r.n_firms else "no rating history",
        axis=1,
    )
    frame["filed"] = pd.to_datetime(frame["filing_date"])
    return frame


@st.cache_data(show_spinner=False)
def load_raw_runs() -> dict[str, dict]:
    path = BACKTEST_DIR / "raw_runs.json"
    if not path.exists():
        return {}
    return {entry["ticker"]: entry for entry in json.loads(path.read_text(encoding="utf-8"))}


def _hits(series: pd.Series) -> tuple[int, int]:
    scored = series.dropna().astype(bool)
    return int(scored.sum()), int(len(scored))


def _pct(num: int, den: int) -> str:
    return f"{round(100 * num / den + 1e-9):d}%" if den else "n/a"  # 62.5 -> 63, not banker's 62


def _dot(stance: str) -> str:
    color = STANCE_COLOR_MAP.get(stance, NO_DATA_COLOR)
    return f'<span class="tr-dot" style="background:{color}"></span>'


def _chip(ticker: str, ret: float, stance: str) -> str:
    return f'<span class="tr-chip">{_dot(stance)}{escape(ticker)} {ret:+.1f}%</span>'


# ── Header + KPIs ────────────────────────────────────────────────────────
df = load_comparison()
agent_hits, agent_n = _hits(df["agent_hit"])
street_hits, street_n = _hits(df["street_hit"])
comparable = df[df["agrees"].notna()]
n_agree = int(comparable["agrees"].astype(bool).sum())
n_no_street = len(df) - len(comparable)
first, last = df["filed"].min(), df["filed"].max()

st.title("Track Record")
st.markdown(
    f'<p class="tr-lede">The agent was backtested on <b>{len(df)} large-cap 10-Qs</b> '
    f"(filed {first:%b %Y} – {last:%b %Y}), each run using only the financial data and prices "
    "available on the filing date. Each call is scored two ways: against what the stock did over "
    "the next 90 days, and against what Wall Street analysts were saying <i>on that same date</i>.</p>",
    unsafe_allow_html=True,
)

kpis = [
    ("Filings tested", f"{len(df)}", "one 10-Q per company, sector-diversified sample"),
    ("Agent hit rate", _pct(agent_hits, agent_n), f"{agent_hits} of {agent_n} Bullish/Bearish calls"),
    ("Wall Street hit rate", _pct(street_hits, street_n), f"{street_hits} of {street_n} consensus calls, same filings"),
    ("Agreed with the Street", f"{n_agree} of {len(comparable)}",
     f"{n_no_street} filings had no rating history" if n_no_street else "all filings had coverage"),
]
st.markdown(
    '<div class="tr-kpis">' + "".join(
        f'<div class="tr-kpi"><div class="tr-kpi-label">{label}</div>'
        f'<div class="tr-kpi-value">{value}</div><div class="tr-kpi-sub">{sub}</div></div>'
        for label, value, sub in kpis
    ) + "</div>",
    unsafe_allow_html=True,
)

# ── Where it disagreed ───────────────────────────────────────────────────
st.subheader("Where it disagreed with Wall Street")
directional = {"Bullish", "Bearish"}
contrarian = comparable[(~comparable["agrees"].astype(bool)) & comparable["agent_stance"].isin(directional)]
cautious = comparable[(comparable["agent_stance"] == "Neutral") & comparable["street_stance"].isin(directional)]
c_hits, c_n = _hits(contrarian["agent_hit"])
s_hits, s_n = _hits(cautious["street_hit"])

callouts = [
    (
        STANCE_COLOR_MAP["Bullish"], "It made a different directional call",
        f"Agent right {c_hits} of {c_n}",
        "It took a Bullish or Bearish stance the consensus didn't share.",
        contrarian,
    ),
    (
        STANCE_COLOR_MAP["Neutral"], "It stayed Neutral while the Street picked a side",
        f"Street right {s_hits} of {s_n}",
        "Mostly the agent holding back on names analysts rated Buy, and those names kept rising.",
        cautious,
    ),
]
st.markdown(
    '<div class="tr-callouts">' + "".join(
        f'<div class="tr-callout" style="--tr-edge:{edge}"><h4>{title}</h4>'
        f'<div class="tr-big">{big}</div><p>{body}</p><div class="tr-chips">'
        + "".join(_chip(r.ticker, r.actual_return_pct, r.agent_stance) for r in group.itertuples())
        + "</div></div>"
        for edge, title, big, body, group in callouts
    ) + "</div>",
    unsafe_allow_html=True,
)
st.caption("Chips show each stock's actual 90-day return after filing; the dot is the agent's call.")

# ── Every call vs. what happened ─────────────────────────────────────────
st.subheader("Every call vs. what happened")
color_by = st.segmented_control(
    "Color bars by", ["Agent's call", "Wall Street consensus"], default="Agent's call",
    key="tr_color_by",
) or "Agent's call"
stance_field = "agent_stance" if color_by == "Agent's call" else "street_label"

bar_df = df.assign(stance_shown=df[stance_field])
legend_domain = STANCES + ([NO_DATA] if (bar_df["stance_shown"] == NO_DATA).any() else [])
bars = (
    alt.Chart(bar_df)
    .mark_bar(cornerRadiusEnd=4, height={"band": 0.72})
    .encode(
        y=alt.Y("ticker:N", sort=alt.EncodingSortField("actual_return_pct", order="descending"),
                title=None, axis=alt.Axis(labelFontWeight="bold", ticks=False, domain=False)),
        x=alt.X("actual_return_pct:Q", title="Stock return, 90 days after filing (%)",
                axis=alt.Axis(grid=True, gridOpacity=0.35, tickCount=6)),
        color=alt.Color(
            "stance_shown:N",
            scale=alt.Scale(domain=legend_domain, range=[*STANCE_COLORS, NO_DATA_COLOR][:len(legend_domain)]),
            legend=alt.Legend(title=color_by, orient="top", direction="horizontal"),
        ),
        tooltip=[
            alt.Tooltip("company:N", title="Company"),
            alt.Tooltip("filing_date:N", title="10-Q filed"),
            alt.Tooltip("agent_stance:N", title="Agent"),
            alt.Tooltip("street_label:N", title="Wall Street"),
            alt.Tooltip("street_mix:N", title="Analyst ratings"),
            alt.Tooltip("implied_upside_pct:Q", title="Street target upside (%)", format="+.1f"),
            alt.Tooltip("actual_return_pct:Q", title="Actual 90d return (%)", format="+.1f"),
        ],
    )
)
zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeWidth=1, opacity=0.6).encode(x="x:Q")
st.altair_chart((bars + zero).properties(height=34 * len(df)), width="stretch")

# ── Average return by call, side by side ─────────────────────────────────
st.subheader("Did the calls sort winners from losers?")
st.caption(
    "Average 90-day return for each stance. A useful signal would show Bullish above Neutral above Bearish."
)


def _avg_by_stance(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    grouped = frame[frame[field].isin(STANCES)].groupby(field)["actual_return_pct"].agg(["mean", "count"])
    grouped = grouped.reindex(STANCES).dropna().reset_index()
    grouped.columns = ["stance", "avg_return", "n"]
    grouped["label"] = grouped.apply(lambda r: f"{r.avg_return:+.1f}%  (n={int(r.n)})", axis=1)
    return grouped


def _avg_chart(frame: pd.DataFrame, title: str, y_domain: list[float]) -> alt.LayerChart:
    # Same x categories and y range on both charts so the pair reads side by side.
    base = alt.Chart(frame).encode(
        x=alt.X("stance:N", scale=alt.Scale(domain=STANCES), title=None,
                axis=alt.Axis(labelAngle=0, ticks=False)),
        y=alt.Y("avg_return:Q", title="Avg 90-day return (%)",
                scale=alt.Scale(domain=y_domain), axis=alt.Axis(gridOpacity=0.35)),
    )
    columns = base.mark_bar(cornerRadiusEnd=4, width={"band": 0.55}).encode(
        color=alt.Color("stance:N", scale=STANCE_SCALE, legend=None),
        tooltip=[alt.Tooltip("stance:N", title="Stance"), alt.Tooltip("n:Q", title="Filings"),
                 alt.Tooltip("avg_return:Q", title="Avg return (%)", format="+.2f")],
    )
    above = base.transform_filter(alt.datum.avg_return >= 0).mark_text(
        dy=-9, fontWeight="bold", color=TEXT_COLOR).encode(text="label:N")
    below = base.transform_filter(alt.datum.avg_return < 0).mark_text(
        dy=11, fontWeight="bold", color=TEXT_COLOR).encode(text="label:N")
    return (columns + above + below).properties(title=title, height=260)


agent_avg = _avg_by_stance(df, "agent_stance")
street_avg = _avg_by_stance(df, "street_stance")
all_avgs = pd.concat([agent_avg["avg_return"], street_avg["avg_return"]])
y_domain = [min(0.0, all_avgs.min()) - 3, max(0.0, all_avgs.max()) + 3]  # headroom for labels
left, right = st.columns(2)
with left:
    st.altair_chart(_avg_chart(agent_avg, "The agent's calls", y_domain), width="stretch")
with right:
    st.altair_chart(_avg_chart(street_avg, "Wall Street consensus", y_domain), width="stretch")

# ── Agreement matrix ─────────────────────────────────────────────────────
st.subheader("Agent vs. Street, call by call")
grid = pd.MultiIndex.from_product([STANCES, STANCES], names=["agent_stance", "street_stance"])
matrix = (
    comparable.groupby(["agent_stance", "street_stance"]).size().rename("count")
    .reindex(grid, fill_value=0).reset_index()
)
matrix["tickers"] = matrix.apply(
    lambda r: ", ".join(comparable[(comparable.agent_stance == r.agent_stance)
                                   & (comparable.street_stance == r.street_stance)]["ticker"]) or "none",
    axis=1,
)
heat_base = alt.Chart(matrix).encode(
    x=alt.X("street_stance:N", sort=STANCES, title="Wall Street consensus",
            axis=alt.Axis(orient="top", labelAngle=0, ticks=False, domain=False)),
    y=alt.Y("agent_stance:N", sort=STANCES, title="Agent's call", axis=alt.Axis(ticks=False, domain=False)),
)
heat = heat_base.mark_rect(cornerRadius=6, stroke=CELL_GAP_COLOR, strokeWidth=3).encode(
    color=alt.Color("count:Q", scale=alt.Scale(range=["#eef4fc", "#1c5cab"]), legend=None),
    tooltip=[alt.Tooltip("agent_stance:N", title="Agent"), alt.Tooltip("street_stance:N", title="Street"),
             alt.Tooltip("count:Q", title="Filings"), alt.Tooltip("tickers:N", title="Tickers")],
)
heat_text = heat_base.mark_text(fontSize=18, fontWeight="bold").encode(
    text="count:Q",
    color=alt.condition(alt.datum.count >= 4, alt.value("white"), alt.value("#1f2937")),
)
m_left, m_right = st.columns([3, 2])
with m_left:
    st.altair_chart((heat + heat_text).properties(height=280), width="stretch")
with m_right:
    st.markdown(
        f"The diagonal is agreement: **{n_agree} of {len(comparable)}** filings. "
        "Wall Street was Bullish on almost everything, which is normal for large caps: sell-side "
        "ratings lean heavily toward Buy. So most disagreements are the agent being *more cautious*, "
        "not more optimistic."
    )
    if n_no_street:
        missing = ", ".join(df[df["agrees"].isna()]["ticker"])
        st.markdown(
            f'<p class="tr-caveat">Left out: {missing}. The free rating feed (yfinance) has no '
            "rating history for them in the year before the filing.</p>",
            unsafe_allow_html=True,
        )

# ── Full table ───────────────────────────────────────────────────────────
st.subheader("All filings")


def _mark(value) -> str:
    if pd.isna(value):
        return "—"
    return "✓" if bool(value) else "✗"


table = pd.DataFrame({
    "Ticker": df["ticker"],
    "Company": df["company"],
    "10-Q filed": df["filed"].dt.date,
    "Agent": df["agent_stance"],
    "Wall Street": df["street_label"],
    "Analyst ratings": df["street_mix"],
    "Street target upside": df["implied_upside_pct"],
    "Actual 90d return": df["actual_return_pct"],
    "Agent right?": df["agent_hit"].map(_mark),
    "Street right?": df["street_hit"].map(_mark),
})
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "Street target upside": st.column_config.NumberColumn(
            format="%+.1f%%", help="Mean analyst price target vs. price on the filing date"),
        "Actual 90d return": st.column_config.NumberColumn(format="%+.1f%%"),
        "Agent right?": st.column_config.TextColumn(help="Neutral calls aren't scored (—)"),
    },
)

# ── Agent reasoning ──────────────────────────────────────────────────────
runs = load_raw_runs()
st.subheader("Read the agent's reasoning")
saved = [t for t in df["ticker"] if t in runs]
if saved:
    st.caption(
        f"Full write-ups are saved for {len(saved)} of {len(df)} filings; the rest were logged as "
        "stance + return only and fill in on the next backtest run."
    )
    pick = st.selectbox("Filing", saved, format_func=lambda t: f"{t} — 10-Q filed {runs[t]['filing_date']}")
    row = df[df["ticker"] == pick].iloc[0]
    st.markdown(
        f"{_dot(row.agent_stance)} **Agent: {row.agent_stance}** &nbsp;·&nbsp; "
        f"{_dot(row.street_label)} **Street: {row.street_label}** ({row.street_mix}) &nbsp;·&nbsp; "
        f"**Actual: {row.actual_return_pct:+.1f}%**",
        unsafe_allow_html=True,
    )
    entry = runs[pick]
    tabs = st.tabs(["Final stance", *[s["title"] for s in entry["steps"]]])
    for tab, text in zip(tabs, [entry["recommendation"], *[s["output"] for s in entry["steps"]]]):
        with tab:
            # '$' escaped so paired dollar amounts don't render as LaTeX (same fix as the Analyze page)
            st.markdown(text.replace("$", "\\$"))
else:
    st.caption("No saved write-ups yet.")

# ── Method ───────────────────────────────────────────────────────────────
with st.expander("How this was measured, and what it can't tell you"):
    st.markdown(
        """
**Agent call.** For each company the pipeline analyzed the 10-Q from two quarters back, using only
XBRL facts the company had filed by that date and the stock price as of that date. No later data
leaks in. The final stance (Bullish / Neutral / Bearish) is read from the agent's own final step.

**Wall Street consensus.** Rebuilt from yfinance's dated analyst rating history. For each broker,
the most recent rating *before* the filing date (within the prior 12 months) counts. Each rating maps
to +1 (Buy / Overweight / Outperform …), 0 (Hold / Neutral / Equal-Weight …), or −1 (Sell /
Underweight …). The average above +⅓ is Bullish, below −⅓ Bearish, otherwise Neutral. At least 3
firms are needed. "Street target upside" is the mean price target vs. the filing-date price.

**Scoring.** A Bullish or Bearish call is right if the stock moved that way over the next 90
calendar days. Neutral calls are shown, not scored.

**Limits.**
- 20 filings from one quarter is a small sample. A few calls either way would swing these percentages
  by 10+ points, so none of this shows real predictive power.
- The window (late 2025 to early 2026) was mostly a rising market, which favors anyone who leans Bullish.
- The free rating feed has gaps (two companies had no recent ratings) and may miss some brokers.
- Only one 10-Q per company, no transaction costs, no benchmark adjustment (returns aren't vs. the S&P 500).
"""
    )
    st.caption("Reproduce: `python scripts/backtest_accuracy.py` then `python scripts/analyst_comparison.py`.")
