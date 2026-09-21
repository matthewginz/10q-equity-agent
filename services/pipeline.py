"""
The multi-step analysis pipeline: real SEC filing data -> five sequential
LLM steps, each one grounded in the previous steps' actual output (not five
independent prompts against the same raw context) -> a final equity
recommendation.

Adapted from the original amazon-10q-agent's 5-prompt structure, but
generalized off standardized XBRL ratios instead of one hardcoded Excel
file, and extended with an explicit final recommendation step the original
never had.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.market_data import MarketSnapshot, format_market_snapshot
from services.gemini_client import generate

_FMT_SYSTEM = (
    "You are an equity research analyst. Be concrete and numbers-grounded. "
    "Never invent a figure that isn't in the provided context -- if something "
    "needed isn't available, say so explicitly rather than guessing. "
    "Do not use the words 'buy' or 'sell' as a directive -- use a "
    "qualitative stance (Bullish / Neutral / Bearish) with reasoning instead."
)


@dataclass
class StepResult:
    title: str
    output: str


@dataclass
class AnalysisRun:
    company: str
    ticker: str
    report_date: str
    steps: list[StepResult] = field(default_factory=list)
    recommendation: str = ""


def _fmt_ratios(ratios: dict) -> str:
    lines = []
    for section, values in ratios.items():
        if section == "data_gaps":
            continue
        lines.append(f"\n{section.upper()}:")
        for k, v in values.items():
            if v is None:
                lines.append(f"  {k}: not available in this filing's XBRL data")
            elif isinstance(v, float) and abs(v) < 5:
                lines.append(f"  {k}: {v:.2%}" if "margin" in k or "yoy" in k or "ratio" in k and abs(v) < 3 else f"  {k}: {v:.3f}")
            else:
                lines.append(f"  {k}: {v:,.1f}")
    if ratios.get("data_gaps"):
        lines.append(f"\nNOTE -- these metrics were not found tagged in this filing's XBRL data: {', '.join(ratios['data_gaps'])}")
    return "\n".join(lines)


def run_analysis(
    company: str, ticker: str, report_date: str, ratios: dict, filing_text: str,
    market: MarketSnapshot | None = None, on_step=None, on_step_done=None,
) -> AnalysisRun:
    """on_step(step_number, title), called just before each step runs, and
    on_step_done(step_number, title, output), called right after -- together
    they let a caller (the Streamlit UI) show live progress AND a recap of
    what each step actually found through a pipeline that can take the
    better part of a minute end to end. Both purely optional, default to a
    no-op so this stays usable from a plain script."""
    if on_step is None:
        on_step = lambda *_: None
    if on_step_done is None:
        on_step_done = lambda *_: None
    run = AnalysisRun(company=company, ticker=ticker, report_date=report_date)
    ratio_block = _fmt_ratios(ratios)
    # edgar_client.fetch_filing_text already returns plain text anchored on
    # the MD&A section heading (not raw HTML from a fixed byte offset), so
    # no further slicing is needed here -- just bound it for the prompt.
    filing_excerpt = filing_text[:30_000]

    # Step 1 -- quantitative snapshot from the real, structured ratios.
    on_step(1, "Quantitative Snapshot")
    s1 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Company: {company} ({ticker}), 10-Q for period ending {report_date}.\n\n"
            f"Computed financial ratios (from this filing's actual SEC XBRL data):\n{ratio_block}\n\n"
            "Write a quantitative snapshot: profitability trend, liquidity/leverage posture, "
            "and the most notable YoY moves. Cite the actual numbers. If a metric is marked "
            "unavailable, say plainly that this filer didn't tag it rather than estimating it."
        ),
        max_output_tokens=800,
    )
    run.steps.append(StepResult("1. Quantitative Snapshot", s1))
    on_step_done(1, "Quantitative Snapshot", s1)

    # Step 2 -- MD&A / risk-factor synthesis from the real filing text.
    on_step(2, "Risk & MD&A Synthesis")
    s2 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Below is an excerpt of {company}'s actual 10-Q filing text (HTML, may include markup noise "
            f"-- read through it for the MD&A and Risk Factors content):\n\n{filing_excerpt}\n\n"
            "Identify and rank the top 3 risks disclosed, and summarize management's stated priorities "
            "or mitigation strategy for each, in a markdown table (Risk | Evidence from filing | Priority)."
        ),
        max_output_tokens=800,
    )
    run.steps.append(StepResult("2. Risk & MD&A Synthesis", s2))
    on_step_done(2, "Risk & MD&A Synthesis", s2)

    # Step 3 -- segment/geographic detail and forward-looking guidance, pulled
    # directly from the filing text. Step 1 is necessarily consolidated-only
    # (that's all XBRL ratios give you); when one line item swings hard (e.g.
    # operating income down while revenue is up), the filing's own segment
    # breakdown is usually the fastest way to see WHERE that came from --
    # without this, the final reconciliation step has nothing concrete to
    # point to beyond "the numbers don't line up."
    on_step(3, "Segment & Forward-Looking Detail")
    s3 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Below is the same excerpt of {company}'s 10-Q filing text used in the prior step:\n\n{filing_excerpt}\n\n"
            "Extract only what the filing ACTUALLY discloses -- never estimate or infer a figure it doesn't state:\n"
            "1. Segment or geographic revenue/margin breakdown, if the filing reports results by segment or "
            "region -- name each segment/region and its stated performance. If the filer only reports one "
            "consolidated segment, say so explicitly rather than inventing a breakdown.\n"
            "2. Any forward-looking guidance or outlook management gave (revenue/margin targets, expected "
            "demand, stated expectations for the next quarter/year). If none is given, say so.\n"
            "3. Material recent developments disclosed in the filing (M&A, restructuring, discontinued "
            "operations, material legal proceedings, subsequent events) -- name each with the filing's own "
            "stated detail, not a generalization."
        ),
        max_output_tokens=700,
    )
    run.steps.append(StepResult("3. Segment & Forward-Looking Detail", s3))
    on_step_done(3, "Segment & Forward-Looking Detail", s3)

    # Step 4 -- consistency check: does the qualitative story match the numbers?
    on_step(4, "Narrative-vs-Numbers Consistency Check")
    s4 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Quantitative snapshot (step 1):\n{s1}\n\n"
            f"Risk/MD&A synthesis (step 2):\n{s2}\n\n"
            f"Segment & forward-looking detail (step 3):\n{s3}\n\n"
            "Cross-check these: does management's narrative in the filing (steps 2-3) match what the actual "
            "numbers show (step 1)? Flag any place where the tone of the disclosure, or a segment's stated "
            "performance, seems more optimistic or more cautious than the consolidated ratios support. If "
            "nothing material is available to check, say so."
        ),
        max_output_tokens=600,
    )
    run.steps.append(StepResult("4. Narrative-vs-Numbers Consistency Check", s4))
    on_step_done(4, "Narrative-vs-Numbers Consistency Check", s4)

    # Step 5 -- capital allocation / sustainability read.
    on_step(5, "Capital Allocation & Sustainability")
    s5 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"Quantitative snapshot:\n{s1}\n\n"
            "Focusing only on cash flow, capex, and leverage figures in that snapshot: assess whether "
            "the company's capital allocation looks sustainable at current free cash flow generation, "
            "and whether leverage is a real constraint. If cash flow or capex data wasn't available, "
            "say so explicitly instead of assessing this blind."
        ),
        max_output_tokens=500,
    )
    run.steps.append(StepResult("5. Capital Allocation & Sustainability", s5))
    on_step_done(5, "Capital Allocation & Sustainability", s5)

    # Step 6 -- final recommendation, synthesizing steps 1-5 AND real market
    # context, not raw data. The market block is what lets this step talk
    # about today's actual price and sentiment instead of only the filing's
    # own fundamentals in isolation -- without it, a well-reasoned bearish
    # read on bad fundamentals (e.g. negative margins) has no way to
    # reconcile against the fact the stock may have already priced that in,
    # or already rallied past it. Step 3's segment/guidance detail is what
    # lets the "why it's at this price" reconciliation point at something
    # concrete (a specific segment or a stated outlook) instead of just the
    # consolidated numbers.
    on_step(6, "Final Equity Stance")
    market_block = format_market_snapshot(market)
    s6 = generate(
        system=_FMT_SYSTEM,
        user=(
            f"You have five prior analysis steps for {company} ({ticker}):\n\n"
            f"1) Quantitative snapshot:\n{s1}\n\n"
            f"2) Risk/MD&A synthesis:\n{s2}\n\n"
            f"3) Segment & forward-looking detail:\n{s3}\n\n"
            f"4) Consistency check:\n{s4}\n\n"
            f"5) Capital allocation read:\n{s5}\n\n"
            f"Market context:\n{market_block}\n\n"
            "Begin your response with EXACTLY this heading as the first line, nothing before it: "
            "'### Final Equity Research Stance: Bullish' or '### Final Equity Research Stance: Neutral' or "
            "'### Final Equity Research Stance: Bearish' -- pick the one word that matches your call. "
            "Then, below that heading, cover in order:\n"
            "1. Sentiment: in one paragraph, what the market's sentiment on this stock looks like right now, "
            "given everything above.\n"
            "2. Why it's at this price: if any single metric in the steps above looks alarming taken alone "
            "(e.g. a sharp operating income or margin decline), name the specific other components from those "
            "same steps that offset it and still support today's valuation -- pointing at a specific segment "
            "or stated outlook from step 3 where relevant -- or say plainly if nothing does and the metric is "
            "a real, unoffset red flag.\n"
            "3. The single biggest supporting factor and the single biggest risk to the thesis.\n"
            "4. Your recommendation as the analyst, and the reasoning behind it.\n"
            "Ground every claim in the five steps and the market context above -- do not introduce new figures, "
            "and do not speculate about market efficiency or whether news is 'priced in' in the abstract; just "
            "explain the picture as it stands and make the call."
        ),
        max_output_tokens=700,
    )
    run.recommendation = s6
    on_step_done(6, "Final Equity Stance", s6)
    return run


def compare_stances(
    company1: str, ticker1: str, run1: AnalysisRun,
    company2: str, ticker2: str, run2: AnalysisRun,
) -> str:
    """One additional grounded LLM call that actually weighs two completed
    AnalysisRuns against each other. run_analysis produces two fully
    independent stances with no comparative judgment between them, which
    defeats the point of running it in compare mode -- this closes that
    gap without re-deriving anything: grounded only in each run's own
    quantitative snapshot (step 1) and final stance, so it can't introduce
    a number neither run already surfaced."""
    snap1 = next((s.output for s in run1.steps if s.title.startswith("1.")), "")
    snap2 = next((s.output for s in run2.steps if s.title.startswith("1.")), "")
    return generate(
        system=_FMT_SYSTEM,
        user=(
            "Two completed equity research analyses, each grounded in that company's real, latest 10-Q:\n\n"
            f"=== {company1} ({ticker1}) ===\n"
            f"Quantitative snapshot: {snap1}\n\n"
            f"Final stance: {run1.recommendation}\n\n"
            f"=== {company2} ({ticker2}) ===\n"
            f"Quantitative snapshot: {snap2}\n\n"
            f"Final stance: {run2.recommendation}\n\n"
            "Compare these two directly, head-to-head, across growth, profitability, and risk. Then give one "
            "explicit verdict: if you could only hold one of these two names, which one, and why, in one "
            "paragraph. Ground every claim in the material above -- do not introduce new figures."
        ),
        max_output_tokens=600,
    )
