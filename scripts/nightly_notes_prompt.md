You are reviewing last night's automated backtest run for the 10q-equity-agent
project (a Streamlit app that turns SEC 10-Q filings into a Bullish/Neutral/Bearish
call, plus a Track Record page that backtests those calls against real returns
and Wall Street consensus). You can only read files. Do not suggest running anything.

Read:
- docs/backtest/progress.json and docs/backtest/results.csv (what's scored so far)
- docs/backtest/analyst_comparison.csv (agent vs. Street)
- the last ~150 lines of logs/backtest.log (tonight's run: failures, retirements, publish result)
- notes/nightly_notes.md (earlier entries -- do NOT repeat an idea or question already there)
- whatever app code you need (app.py, views/, ui/, core/, services/, scripts/)

Then output ONLY one markdown section, nothing before or after it, in exactly this shape:

## <YYYY-MM-DD, today's date>
**Run:** one line -- filings scored tonight, total done / in window, publish pushed or not.
**Problems:** one line -- errors, retired models, or failed filings from the log; "none" if clean.
**Numbers:** one line -- agent vs. Street hit rate (raw and vs. SPY) with scored-call counts. Say plainly if the gap is not yet meaningful.
**Idea (site/UI):** one concrete, new improvement to the website, grounded in what you read (name the page/file it touches). 2-3 sentences.
**Idea (backtest):** one concrete, new way to make the backtest more trustworthy or informative. 2-3 sentences.
**Question for Matthew:** one new question whose answer would change what gets built next -- something only he can answer (goals, audience, money, taste), not something you could look up.

Keep the whole section under 170 words. Plain, specific language; no hype.
