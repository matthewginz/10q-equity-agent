"""
Track Record page -- how the agent's backtested stances held up, against
both what the stock actually did and what Wall Street was saying on the
same date. Built for a sample of hundreds of filings: every hit rate is
shown with a 95% interval and a coin-flip test, never as a bare percentage.

Reads only committed files (no network, no LLM):
  docs/backtest/analyst_comparison.csv  <- scripts/analyst_comparison.py
  docs/backtest/raw_runs.json           <- scripts/backtest_accuracy.py
  docs/backtest/progress.json           <- scripts/backtest_accuracy.py

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

from core.benchmarks import baselines, long_short
from core.stats import HitRate, hit_rate

BACKTEST_DIR = Path(__file__).resolve().parent.parent / "docs" / "backtest"
STANCES = ["Bullish", "Neutral", "Bearish"]
DIRECTIONAL = {"Bullish", "Bearish"}
STANCE_COLORS = ["#2a78d6", "#8b8b87", "#dc2626"]
STANCE_SCALE = alt.Scale(domain=STANCES, range=STANCE_COLORS)
STANCE_COLOR_MAP = dict(zip(STANCES, STANCE_COLORS))
NO_DATA = "No data"
NO_DATA_COLOR = "#d4d4d0"
SIGNIFICANCE = 0.05
# Found live: Streamlit 1.64 ignores a layered chart's fixed spec height and squashes it to
# about half. Dot-interval charts size per row (alt.Step); the rest pass a pixel height.
RATE_ROW_STEP = 34
MAX_CHIPS = 10
# Chart text follows the active theme's ink, never a series color.
_IS_DARK = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
TEXT_COLOR = "#e5e7eb" if _IS_DARK else "#1f2937"
MUTED_COLOR = "#9ca3af" if _IS_DARK else "#6b7280"
CELL_GAP_COLOR = "#0e1117" if _IS_DARK else "#ffffff"

RAW, VS_MARKET = "Raw return", "Return vs. S&P 500"
SCORING = {
    RAW: {"ret": "actual_return_pct", "agent": "agent_hit", "street": "street_hit",
          "axis": "Stock return, 90 days after filing (%)"},
    VS_MARKET: {"ret": "excess_return_pct", "agent": "agent_hit_vs_market", "street": "street_hit_vs_market",
                "axis": "Return minus S&P 500, 90 days after filing (pts)"},
}

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 1400px; }
.tr-lede { font-size: 1.02rem; opacity: 0.8; max-width: 62rem; margin: -0.5rem 0 1rem; }
.tr-progress { border: 1px solid rgba(128,128,128,0.25); border-radius: 0.6rem; padding: 0.6rem 0.9rem;
    margin: 0 0 1.25rem; font-size: 0.9rem; }
.tr-bar { height: 6px; border-radius: 999px; background: rgba(128,128,128,0.2); margin-top: 0.45rem; overflow: hidden; }
.tr-bar > span { display: block; height: 100%; background: #2a78d6; }
.tr-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0.9rem; margin-bottom: 1.5rem; }
.tr-kpi { padding: 1rem 1.15rem; border-radius: 0.75rem;
    border: 1px solid rgba(128,128,128,0.22); background: rgba(128,128,128,0.05); }
.tr-kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.65; font-weight: 600; }
.tr-kpi-value { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.15; margin-top: 0.2rem;
    font-variant-numeric: tabular-nums; }
.tr-kpi-sub { font-size: 0.84rem; opacity: 0.72; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }
.tr-sig { display: inline-block; margin-top: 0.5rem; font-size: 0.78rem; font-weight: 600; padding: 0.1rem 0.5rem;
    border-radius: 999px; border: 1px solid rgba(128,128,128,0.35); }
.tr-callouts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 0.9rem; margin: 0.25rem 0 0.5rem; }
.tr-callout { padding: 1rem 1.2rem; border-radius: 0.75rem; border: 1px solid rgba(128,128,128,0.22);
    border-left: 4px solid var(--tr-edge); }
.tr-callout h4 { margin: 0 0 0.35rem; font-size: 1rem; padding: 0; }
.tr-callout .tr-big { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.tr-callout p { margin: 0.35rem 0 0; font-size: 0.9rem; opacity: 0.85; }
.tr-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.6rem; }
.tr-chip { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.8rem; font-weight: 600;
    padding: 0.15rem 0.55rem; border-radius: 999px; border: 1px solid rgba(128,128,128,0.25);
    font-variant-numeric: tabular-nums; }
.tr-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.tr-caveat { font-size: 0.85rem; opacity: 0.75; border-left: 3px solid rgba(128,128,128,0.5); padding-left: 0.75rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Data ─────────────────────────────────────────────────────────────────
def _quarter(ts: pd.Timestamp) -> str:
    return f"{ts.year} Q{(ts.month - 1) // 3 + 1}"


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    frame = pd.read_csv(BACKTEST_DIR / "analyst_comparison.csv")
    frame["street_label"] = frame["street_stance"].replace({"Insufficient": NO_DATA})
    frame["street_mix"] = frame.apply(
        lambda r: f"{r.n_buy} Buy · {r.n_hold} Hold · {r.n_sell} Sell" if r.n_firms else "no rating history",
        axis=1,
    )
    frame["filed"] = pd.to_datetime(frame["filing_date"])
    frame["quarter"] = frame["filed"].map(_quarter)
    return frame


@st.cache_data(show_spinner=False)
def load_raw_runs() -> dict[tuple[str, str], dict]:
    path = BACKTEST_DIR / "raw_runs.json"
    if not path.exists():
        return {}
    return {(e["ticker"], e["filing_date"]): e for e in json.loads(path.read_text(encoding="utf-8"))}


@st.cache_data(show_spinner=False)
def load_progress() -> dict | None:
    path = BACKTEST_DIR / "progress.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _rate(series: pd.Series) -> HitRate:
    scored = series.dropna().astype(bool)
    return hit_rate(int(scored.sum()), int(len(scored)))


# ── Small HTML builders ──────────────────────────────────────────────────
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{round(100 * value + 1e-9):d}%"


def _ci(r: HitRate) -> str:
    return "" if r.ci_low is None else f"95% CI {_pct(r.ci_low)}–{_pct(r.ci_high)}"


def _sig_badge(r: HitRate) -> str:
    if r.p_value is None:
        return ""
    if r.p_value < SIGNIFICANCE:
        shown = "p < 0.001" if r.p_value < 0.001 else f"p = {r.p_value:.3f}"
        return f'<span class="tr-sig">✓ Beats a coin flip ({shown})</span>'
    return f'<span class="tr-sig">≈ Could still be luck (p = {r.p_value:.2f})</span>'


def _dot(stance: str) -> str:
    return f'<span class="tr-dot" style="background:{STANCE_COLOR_MAP.get(stance, NO_DATA_COLOR)}"></span>'


def _chip(ticker: str, when: str, ret: float, stance: str) -> str:
    return f'<span class="tr-chip">{_dot(stance)}{escape(ticker)} {when[:7]} {ret:+.1f}%</span>'


def _kpi(label: str, value: str, sub: str, badge: str = "") -> str:
    return (f'<div class="tr-kpi"><div class="tr-kpi-label">{label}</div><div class="tr-kpi-value">{value}</div>'
            f'<div class="tr-kpi-sub">{sub}</div>{badge}</div>')


# ── Sections ─────────────────────────────────────────────────────────────
def render_header(df: pd.DataFrame, progress: dict | None) -> None:
    st.title("Track Record")
    first, last = df["filed"].min(), df["filed"].max()
    st.markdown(
        f'<p class="tr-lede">The agent was backtested on <b>{len(df)} 10-Qs</b> from '
        f"<b>{df['ticker'].nunique()} large-cap companies</b> (filed {first:%b %Y} – {last:%b %Y}), "
        f"answered by {df['model'].nunique()} Google model{'s' if df['model'].nunique() != 1 else ''} "
        "(one model per filing). Each run used only the financial data and prices "
        "available on the filing date. Every call is scored against what the stock did over the next 90 days, "
        "and against what Wall Street analysts were saying <i>on that same date</i>.</p>",
        unsafe_allow_html=True,
    )
    # progress.json is only rewritten at the start and end of a run, so mid-run
    # the scored rows on this page are the fresher count.
    done = max(progress["filings_done"], len(df)) if progress else 0
    if progress and done < progress["filings_in_window"]:
        total = progress["filings_in_window"]
        st.markdown(
            f'<div class="tr-progress"><b>Backtest in progress:</b> {done} of {total} filings scored '
            f"({100 * done / total:.0f}%). The numbers below update as the run continues. "
            f"Last update {progress['updated'].replace('T', ' ')}."
            f'<div class="tr-bar"><span style="width:{100 * done / total:.1f}%"></span></div></div>',
            unsafe_allow_html=True,
        )


def render_kpis(df: pd.DataFrame, cols: dict) -> None:
    agent, street = _rate(df[cols["agent"]]), _rate(df[cols["street"]])
    comparable = df[df["agrees"].notna()]
    agree = hit_rate(int(comparable["agrees"].astype(bool).sum()), len(comparable))
    n_no_street = len(df) - len(comparable)
    cards = [
        _kpi("Filings tested", f"{len(df)}", f"{df['ticker'].nunique()} companies · {df['quarter'].nunique()} quarters"),
        _kpi("Agent hit rate", _pct(agent.rate),
             f"{agent.hits} of {agent.n} Bullish/Bearish calls · {_ci(agent)}", _sig_badge(agent)),
        _kpi("Wall Street hit rate", _pct(street.rate),
             f"{street.hits} of {street.n} consensus calls · {_ci(street)}", _sig_badge(street)),
        _kpi("Agreed with the Street", f"{agree.hits} of {agree.n}",
             f"{_pct(agree.rate)} of filings" + (f" · {n_no_street} had no rating history" if n_no_street else "")),
    ]
    st.markdown('<div class="tr-kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_benchmarks(df: pd.DataFrame, cols: dict) -> None:
    st.subheader("Does it beat simple rules?")
    st.caption("Each caller is scored on its own Bullish/Bearish calls, with a 95% interval. \"Always Bullish\" "
               "is hard to beat in a rising market. Momentum means calling the direction of the stock's prior "
               "90 days. The agent has to beat these rules, not just the coin flip (the dashed line).")
    frame = df if "prior_90d_return_pct" in df else df.assign(prior_90d_return_pct=float("nan"))
    callers = [("Agent", _rate(df[cols["agent"]])), ("Wall Street", _rate(df[cols["street"]]))]
    callers += [(b.name, b.rate) for b in baselines(frame, cols["ret"])]
    table = pd.DataFrame([
        {"caller": name, "rate": r.rate * 100, "lo": r.ci_low * 100, "hi": r.ci_high * 100,
         "n": r.n, "label": f"{r.hits}/{r.n}"}
        for name, r in callers if r.n
    ])
    st.altair_chart(_rate_chart(table, "caller", "Directional hit rate by caller", list(table["caller"])),
                    width="stretch")


def _spread_chart(cohorts: pd.DataFrame) -> alt.LayerChart:
    data = cohorts.assign(sign=cohorts["spread"].map(lambda s: "Bullish" if s >= 0 else "Bearish"))
    base = alt.Chart(data).encode(
        x=alt.X("quarter:N", sort=sorted(data["quarter"]), title=None, axis=alt.Axis(labelAngle=0, ticks=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, width={"band": 0.55}).encode(
        y=alt.Y("spread:Q", title="Long minus short, 90 days (pts)", axis=alt.Axis(gridOpacity=0.35)),
        color=alt.Color("sign:N", scale=STANCE_SCALE, legend=None),
        tooltip=[alt.Tooltip("quarter:N", title="Filing quarter"),
                 alt.Tooltip("n_long:Q", title="Longs (Bullish)"), alt.Tooltip("n_short:Q", title="Shorts (Bearish)"),
                 alt.Tooltip("long_ret:Q", title="Long avg (%)", format="+.2f"),
                 alt.Tooltip("short_ret:Q", title="Short avg (%)", format="+.2f"),
                 alt.Tooltip("spread:Q", title="Spread (pts)", format="+.2f")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=TEXT_COLOR, opacity=0.6).encode(y="y:Q")
    return (bars + zero).properties(title="Long-short return per filing quarter", height=260)


def render_portfolio(df: pd.DataFrame, cols: dict) -> None:
    st.subheader("Would the calls have made money?")
    st.caption("Each filing quarter is one portfolio: equal-weight long every Bullish call, short every Bearish call, "
               "each held 90 days. Neutral calls aren't traded. Quarters with fewer than 5 of either side are "
               "skipped. No trading costs or borrow fees.")
    ls = long_short(df, cols["ret"])
    if ls.cohorts.empty or ls.mean_spread is None:
        st.info("Not enough Bullish and Bearish calls in any one quarter yet. This fills in as the backtest runs.")
        return
    significant = ls.ci_low > 0 or ls.ci_high < 0
    badge = ('<span class="tr-sig">✓ Interval excludes zero</span>' if significant
             else '<span class="tr-sig">≈ Could still be zero</span>')
    cards = [
        _kpi("Long-short spread", f"{ls.mean_spread:+.1f} pts",
             f"per 90 days · 95% CI {ls.ci_low:+.1f} to {ls.ci_high:+.1f}", badge),
        _kpi("Sharpe ratio", "Too early" if ls.sharpe is None else f"{ls.sharpe:.2f}",
             f"annualized from {len(ls.cohorts)} quarterly portfolios"
             + (" (needs 4+)" if ls.sharpe is None else "")),
        _kpi("Winning quarters", f"{ls.winning_cohorts} of {len(ls.cohorts)}", "longs beat shorts"),
        _kpi("Growth of $1", "Too early" if ls.sharpe is None else f"${ls.cumulative[-1]:.2f}",
             "needs 4+ quarterly portfolios" if ls.sharpe is None
             else f"quarters compounded · worst drawdown {ls.max_drawdown:.0%}"),
    ]
    st.markdown('<div class="tr-kpis">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    st.altair_chart(_spread_chart(ls.cohorts), width="stretch", height=330)
    st.caption("A Sharpe ratio from only a few quarters is noisy. Treat the spread's interval as the main evidence.")


def _callout(edge: str, title: str, r: HitRate, who: str, body: str, group: pd.DataFrame, ret_col: str) -> str:
    biggest = group.reindex(group[ret_col].abs().sort_values(ascending=False).index).head(MAX_CHIPS)
    chips = "".join(_chip(x.ticker, x.filing_date, getattr(x, ret_col), x.agent_stance) for x in biggest.itertuples())
    more = f'<span class="tr-chip">+{len(group) - len(biggest)} more</span>' if len(group) > len(biggest) else ""
    return (f'<div class="tr-callout" style="--tr-edge:{edge}"><h4>{title}</h4>'
            f'<div class="tr-big">{who} right {r.hits} of {r.n} ({_pct(r.rate)})</div>'
            f"<p>{body} {_ci(r)}.</p>{_sig_badge(r)}<div class=\"tr-chips\">{chips}{more}</div></div>")


def render_disagreements(df: pd.DataFrame, cols: dict) -> None:
    st.subheader("Where it disagreed with Wall Street")
    comparable = df[df["agrees"].notna()]
    contrarian = comparable[(~comparable["agrees"].astype(bool)) & comparable["agent_stance"].isin(DIRECTIONAL)]
    cautious = comparable[(comparable["agent_stance"] == "Neutral") & comparable["street_stance"].isin(DIRECTIONAL)]
    html = _callout(
        STANCE_COLOR_MAP["Bullish"], "It made a directional call the Street didn't share",
        _rate(contrarian[cols["agent"]]), "Agent", "Bullish or Bearish where the consensus said otherwise.",
        contrarian, cols["ret"],
    ) + _callout(
        STANCE_COLOR_MAP["Neutral"], "It stayed Neutral while the Street picked a side",
        _rate(cautious[cols["street"]]), "Street", "How often the Street's call was right on these.",
        cautious, cols["ret"],
    )
    st.markdown(f'<div class="tr-callouts">{html}</div>', unsafe_allow_html=True)
    st.caption(f"Chips show the biggest moves (up to {MAX_CHIPS}); the dot is the agent's call.")


def render_distribution(df: pd.DataFrame, cols: dict) -> None:
    st.subheader("Every call vs. what happened")
    group_by = st.segmented_control("Group by", ["Agent's call", "Wall Street consensus"],
                                    default="Agent's call", key="tr_group_by") or "Agent's call"
    field = "agent_stance" if group_by == "Agent's call" else "street_label"
    domain = STANCES + ([NO_DATA] if (df[field] == NO_DATA).any() else [])
    colors = [*STANCE_COLORS, NO_DATA_COLOR][:len(domain)]
    base = alt.Chart(df.assign(shown=df[field])).encode(
        y=alt.Y("shown:N", scale=alt.Scale(domain=domain), title=None,
                axis=alt.Axis(labelFontWeight="bold", ticks=False, domain=False)),
    )
    dots = base.transform_calculate(jitter="(random() - 0.5) * 26").mark_circle(size=70, opacity=0.75).encode(
        x=alt.X(f"{cols['ret']}:Q", title=cols["axis"], axis=alt.Axis(gridOpacity=0.35, tickCount=8)),
        yOffset=alt.YOffset("jitter:Q"),
        color=alt.Color("shown:N", scale=alt.Scale(domain=domain, range=colors), legend=None),
        tooltip=[alt.Tooltip("ticker:N", title="Ticker"), alt.Tooltip("filing_date:N", title="10-Q filed"),
                 alt.Tooltip("agent_stance:N", title="Agent"), alt.Tooltip("street_label:N", title="Wall Street"),
                 alt.Tooltip(f"{cols['ret']}:Q", title="Return (%)", format="+.1f"),
                 alt.Tooltip("model:N", title="Model")],
    )
    means = base.mark_tick(thickness=3, size=40, color=TEXT_COLOR).encode(
        x=alt.X(f"mean({cols['ret']}):Q"), tooltip=[alt.Tooltip(f"mean({cols['ret']}):Q", title="Average", format="+.2f")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], opacity=0.6).encode(x="x:Q")
    st.altair_chart((dots + means + zero).properties(height=90 * len(domain)), width="stretch",
                    height=90 * len(domain) + 70)
    st.caption("Each dot is one filing; the dark tick is the average. Hover a dot for details.")


def _avg_with_ci(frame: pd.DataFrame, field: str, ret_col: str) -> pd.DataFrame:
    grouped = frame[frame[field].isin(STANCES)].groupby(field)[ret_col].agg(["mean", "std", "count"])
    grouped = grouped.reindex(STANCES).dropna(subset=["mean"]).reset_index()
    grouped.columns = ["stance", "avg", "std", "n"]
    half = 1.96 * grouped["std"].fillna(0) / grouped["n"].clip(lower=1) ** 0.5
    return grouped.assign(lo=grouped["avg"] - half, hi=grouped["avg"] + half,
                          label=grouped.apply(lambda r: f"{r.avg:+.1f}  (n={int(r.n)})", axis=1))


def _avg_chart(frame: pd.DataFrame, title: str, y_domain: list[float], axis_title: str) -> alt.LayerChart:
    base = alt.Chart(frame).encode(
        x=alt.X("stance:N", scale=alt.Scale(domain=STANCES), title=None, axis=alt.Axis(labelAngle=0, ticks=False)),
    )
    y = alt.Y("avg:Q", title=axis_title, scale=alt.Scale(domain=y_domain), axis=alt.Axis(gridOpacity=0.35))
    bars = base.mark_bar(cornerRadiusEnd=4, width={"band": 0.5}).encode(
        y=y, color=alt.Color("stance:N", scale=STANCE_SCALE, legend=None),
        tooltip=[alt.Tooltip("stance:N"), alt.Tooltip("n:Q", title="Filings"),
                 alt.Tooltip("avg:Q", title="Average", format="+.2f"),
                 alt.Tooltip("lo:Q", title="95% CI low", format="+.2f"), alt.Tooltip("hi:Q", title="95% CI high", format="+.2f")],
    )
    whiskers = base.mark_rule(strokeWidth=2, color=TEXT_COLOR).encode(y="lo:Q", y2="hi:Q")
    labels = base.mark_text(dx=0, dy=-8, fontWeight="bold", color=TEXT_COLOR).encode(y="hi:Q", text="label:N")
    return (bars + whiskers + labels).properties(title=title, height=270)


def render_averages(df: pd.DataFrame, cols: dict) -> None:
    st.subheader("Did the calls sort winners from losers?")
    st.caption("Average return per stance with a 95% interval. Real signal would show Bullish above "
               "Neutral above Bearish, with intervals that don't overlap much.")
    agent = _avg_with_ci(df, "agent_stance", cols["ret"])
    street = _avg_with_ci(df, "street_stance", cols["ret"])
    both = pd.concat([agent[["lo", "hi"]], street[["lo", "hi"]]])
    y_domain = [min(0.0, both["lo"].min()) - 3, max(0.0, both["hi"].max()) + 4]
    left, right = st.columns(2)
    with left:
        st.altair_chart(_avg_chart(agent, "The agent's calls", y_domain, "Average (%)"), width="stretch", height=340)
    with right:
        st.altair_chart(_avg_chart(street, "Wall Street consensus", y_domain, "Average (%)"), width="stretch", height=340)


def _rate_table(df: pd.DataFrame, by: str, hit_col: str) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(by):
        r = _rate(group[hit_col])
        if r.n:
            rows.append({by: key, "rate": r.rate * 100, "lo": r.ci_low * 100, "hi": r.ci_high * 100,
                         "n": r.n, "label": f"{r.hits}/{r.n}"})
    return pd.DataFrame(rows)


def _rate_chart(table: pd.DataFrame, by: str, title: str, sort: list[str] | None = None) -> alt.LayerChart:
    y = alt.Y(f"{by}:N", sort=sort, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontWeight="bold"))
    x_scale = alt.Scale(domain=[0, 100])
    base = alt.Chart(table).encode(y=y)
    whisker = base.mark_rule(strokeWidth=2, color=MUTED_COLOR).encode(x=alt.X("lo:Q", scale=x_scale), x2="hi:Q")
    point = base.mark_circle(size=110, color=TEXT_COLOR, opacity=1).encode(
        x=alt.X("rate:Q", scale=x_scale, title="Directional hit rate (%)", axis=alt.Axis(gridOpacity=0.35)),
        tooltip=[alt.Tooltip(f"{by}:N"), alt.Tooltip("label:N", title="Right"),
                 alt.Tooltip("rate:Q", title="Hit rate (%)", format=".0f"),
                 alt.Tooltip("lo:Q", title="95% CI low", format=".0f"), alt.Tooltip("hi:Q", title="95% CI high", format=".0f")],
    )
    label = base.mark_text(align="left", dx=8, color=TEXT_COLOR).encode(x="hi:Q", text="label:N")
    coin = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(strokeDash=[4, 4], opacity=0.7).encode(x="x:Q")
    return (whisker + point + label + coin).properties(title=title, height=alt.Step(RATE_ROW_STEP))


def render_stability(df: pd.DataFrame, cols: dict) -> None:
    st.subheader("Is it consistent?")
    st.caption("Hit rate with 95% intervals. The dashed line is a coin flip. A real edge should sit right of it "
               "in most quarters and for most models, not just one.")
    by_quarter = _rate_table(df, "quarter", cols["agent"])
    by_model = _rate_table(df, "model", cols["agent"])
    left, right = st.columns(2)
    with left:
        if len(by_quarter):
            st.altair_chart(_rate_chart(by_quarter, "quarter", "By filing quarter", sorted(by_quarter["quarter"])),
                            width="stretch")
    with right:
        if len(by_model):
            st.altair_chart(_rate_chart(by_model, "model", "By model"), width="stretch")


_LEAN = {"Bullish": 1, "Neutral": 0, "Bearish": -1}


def _lean_sentence(comparable: pd.DataFrame) -> str:
    """Which way the disagreements lean, computed rather than asserted."""
    gap = comparable["agent_stance"].map(_LEAN) - comparable["street_stance"].map(_LEAN)
    cautious, bolder = int((gap < 0).sum()), int((gap > 0).sum())
    street_bull = (comparable["street_stance"] == "Bullish").mean() if len(comparable) else 0
    lean = ("mostly the agent being **more cautious** than analysts" if cautious > bolder else
            "mostly the agent being **more optimistic** than analysts" if bolder > cautious else
            "split evenly between more cautious and more optimistic")
    return (f"The Street was Bullish on {street_bull:.0%} of these (sell-side ratings on large caps lean "
            f"toward Buy). Of the disagreements, {cautious} have the agent more cautious and {bolder} more "
            f"optimistic: {lean}.")


def render_matrix(df: pd.DataFrame) -> None:
    st.subheader("Agent vs. Street, call by call")
    comparable = df[df["agrees"].notna()]
    grid = pd.MultiIndex.from_product([STANCES, STANCES], names=["agent_stance", "street_stance"])
    matrix = (comparable.groupby(["agent_stance", "street_stance"]).size().rename("count")
              .reindex(grid, fill_value=0).reset_index())
    base = alt.Chart(matrix).encode(
        x=alt.X("street_stance:N", sort=STANCES, title="Wall Street consensus",
                axis=alt.Axis(orient="top", labelAngle=0, ticks=False, domain=False)),
        y=alt.Y("agent_stance:N", sort=STANCES, title="Agent's call", axis=alt.Axis(ticks=False, domain=False)),
    )
    threshold = max(1, matrix["count"].max() * 0.5)
    heat = base.mark_rect(cornerRadius=6, stroke=CELL_GAP_COLOR, strokeWidth=3).encode(
        color=alt.Color("count:Q", scale=alt.Scale(range=["#eef4fc", "#1c5cab"]), legend=None),
        tooltip=[alt.Tooltip("agent_stance:N", title="Agent"), alt.Tooltip("street_stance:N", title="Street"),
                 alt.Tooltip("count:Q", title="Filings")],
    )
    text = base.mark_text(fontSize=18, fontWeight="bold").encode(
        text="count:Q", color=alt.condition(alt.datum.count >= threshold, alt.value("white"), alt.value("#1f2937")),
    )
    n_agree = int(comparable["agrees"].astype(bool).sum())
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart((heat + text).properties(height=280), width="stretch", height=350)
    with right:
        st.markdown(f"The diagonal is agreement: **{n_agree} of {len(comparable)}** filings. " + _lean_sentence(comparable))
        missing = df[df["agrees"].isna()]
        if len(missing):
            st.markdown(
                f'<p class="tr-caveat">{len(missing)} filings are left out of the comparison: the free rating feed '
                "(yfinance) had no analyst ratings for those companies in the year before the filing.</p>",
                unsafe_allow_html=True,
            )


def _mark(value) -> str:
    return "—" if pd.isna(value) else ("✓" if bool(value) else "✗")


def render_table(df: pd.DataFrame, cols: dict) -> None:
    st.subheader("All filings")
    f1, f2 = st.columns([2, 3])
    query = f1.text_input("Filter by ticker or company", key="tr_query").strip().lower()
    stances = f2.multiselect("Agent's call", STANCES, default=STANCES, key="tr_stances")
    shown = df[df["agent_stance"].isin(stances)]
    if query:
        shown = shown[shown["ticker"].str.lower().str.contains(query) | shown["company"].str.lower().str.contains(query)]
    table = pd.DataFrame({
        "Ticker": shown["ticker"], "Company": shown["company"], "10-Q filed": shown["filed"].dt.date,
        "Agent": shown["agent_stance"], "Wall Street": shown["street_label"], "Analyst ratings": shown["street_mix"],
        "Street target upside": shown["implied_upside_pct"], "90d return": shown["actual_return_pct"],
        "vs S&P 500": shown["excess_return_pct"], "Agent right?": shown[cols["agent"]].map(_mark),
        "Street right?": shown[cols["street"]].map(_mark), "Model": shown["model"],
    }).sort_values("10-Q filed", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
        "Street target upside": st.column_config.NumberColumn(format="%+.1f%%",
                                                              help="Mean analyst price target vs. price on the filing date"),
        "90d return": st.column_config.NumberColumn(format="%+.1f%%"),
        "vs S&P 500": st.column_config.NumberColumn(format="%+.1f", help="Stock return minus SPY return, same window"),
        "Agent right?": st.column_config.TextColumn(help=f"Scored on: {st.session_state.get('tr_scoring', RAW)}. "
                                                         "Neutral calls aren't scored (—)"),
    })
    st.caption(f"{len(table)} of {len(df)} filings shown.")


def render_reasoning(df: pd.DataFrame) -> None:
    runs = load_raw_runs()
    st.subheader("Read the agent's reasoning")
    options = [(r.ticker, r.filing_date) for r in df.sort_values("filed", ascending=False).itertuples()
               if (r.ticker, r.filing_date) in runs]
    if not options:
        st.caption("No saved write-ups yet.")
        return
    pick = st.selectbox("Filing", options, format_func=lambda k: f"{k[0]} — 10-Q filed {k[1]} · {runs[k]['models']}",
                        key="tr_reasoning")
    row = df[(df["ticker"] == pick[0]) & (df["filing_date"] == pick[1])].iloc[0]
    st.markdown(
        f"{_dot(row.agent_stance)} **Agent: {row.agent_stance}** &nbsp;·&nbsp; "
        f"{_dot(row.street_label)} **Street: {row.street_label}** ({row.street_mix}) &nbsp;·&nbsp; "
        f"**Actual: {row.actual_return_pct:+.1f}%** ({row.excess_return_pct:+.1f} vs S&P 500)",
        unsafe_allow_html=True,
    )
    entry = runs[pick]
    tabs = st.tabs(["Final stance", *[s["title"] for s in entry["steps"]]])
    for tab, text in zip(tabs, [entry["recommendation"], *[s["output"] for s in entry["steps"]]]):
        with tab:
            # '$' escaped so paired dollar amounts don't render as LaTeX (same fix as the Analyze page)
            st.markdown(text.replace("$", "\\$"))


def render_method() -> None:
    with st.expander("How this was measured, and what it can't tell you"):
        st.markdown(
            """
**Sample.** The S&P 500, with its membership frozen on Sept. 24, 2026, before these results existed. Every
10-Q each company filed from July 2024 onward counts, as long as 90 days of price history exist after it.
The sample keeps growing as new filings reach 90 days. The newest quarter is scored first, in a fixed
random order within each quarter. Until the run finishes, recent quarters make up more of the results
than older ones, and the by-quarter chart shows that split.

**Agent call.** For each filing the production pipeline saw only XBRL facts filed by that date and the stock
price on that date. It was also told that the filing date was "today". Each filing was answered start to
finish by one model. Several free-tier Google models share the work because of daily limits, and each
row records which one answered. The final stance is read from the agent's own final step.

**Wall Street consensus.** Rebuilt from yfinance's dated analyst ratings: each broker's latest rating
*before* the filing date (within 12 months) maps to +1 / 0 / −1. The average above +⅓ is Bullish, below
−⅓ Bearish, otherwise Neutral (at least 3 firms needed).

**Scoring.** A Bullish or Bearish call is right if the stock moved that way over the next 90 calendar days.
Switch to "Return vs. S&P 500" to score against the market instead, which removes the tailwind of a rising
market. Neutral calls are shown, not scored. Intervals are 95% Wilson intervals. The p-value is a
one-sided exact binomial test against a 50% coin flip.

**Limits.**
- **Model memory.** Gemini may have read about some of these companies' later results in training,
  especially for older filings. The "today is the filing date" instruction limits this but can't rule it
  out, so check the by-quarter chart: an edge that shows up only in older quarters is a warning sign.
- Several models share the work. The by-model chart shows whether one of them behaves differently.
- **Survivorship.** Only companies in the index today are included. Companies that dropped out after a bad
  run are missing, which can flatter Bullish calls.
- The analyst-rating feed has gaps and may miss some brokers. No transaction costs.
"""
        )
        st.caption("Reproduce: `python scripts/backtest_accuracy.py` then `python scripts/analyst_comparison.py`.")


# ── Page ─────────────────────────────────────────────────────────────────
df = load_comparison()
render_header(df, load_progress())
scoring = st.segmented_control("Score calls against", [RAW, VS_MARKET], default=RAW, key="tr_scoring") or RAW
cols = SCORING[scoring]
render_kpis(df, cols)
render_benchmarks(df, cols)
render_portfolio(df, cols)
render_disagreements(df, cols)
render_distribution(df, cols)
render_averages(df, cols)
render_stability(df, cols)
render_matrix(df)
render_table(df, cols)
render_reasoning(df)
render_method()
