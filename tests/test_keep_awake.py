import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import keep_awake as ka

APP = "https://10q-equity-agent.streamlit.app/"


def test_rendered_app_is_ok():
    assert ka.verdict(APP, "10-Q Equity Research Agent Ticker Run analysis", "Run analysis") == "ok"


def test_streamlit_not_found_redirect_fails():
    assert ka.verdict("https://share.streamlit.io/errors/not_found", "", "Run analysis").startswith("FAILED")


def test_access_error_page_fails():
    text = "You do not have access to this app or it does not exist"
    assert ka.verdict(APP, text, "Run analysis").startswith("FAILED")


def test_still_loading_keeps_waiting():
    # the tab title is set before render, so title-only text must not count as up
    assert ka.verdict(APP, "10-Q Equity Research Agent", "Run analysis") is None
