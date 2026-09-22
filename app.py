"""
10-Q Equity Research Agent -- Streamlit entrypoint / page router.

Pages:
  views/analyze.py       -- run the live 6-step analysis on any ticker
  views/track_record.py  -- backtest results vs. actual returns and vs.
                            point-in-time Wall Street consensus

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io, pointed at this repo, app.py as
               the entrypoint, GEMINI_API_KEY set in the app's Secrets.
"""

import streamlit as st

st.set_page_config(page_title="10-Q Equity Research Agent", page_icon="\U0001F4C8", layout="wide")

page = st.navigation(
    [
        st.Page("views/analyze.py", title="Analyze a 10-Q", icon=":material/query_stats:", default=True),
        st.Page("views/track_record.py", title="Track Record", icon=":material/fact_check:"),
    ],
    position="top",
)
page.run()
