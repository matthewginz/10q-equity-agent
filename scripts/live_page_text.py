"""Print the visible text of a page on the deployed Streamlit app.

Used by publish_backtest.py to confirm a push actually reached the live site.
Runs with the website gate's venv (it has Playwright; this project's venv doesn't):
    ../_website-gate/venv/Scripts/python.exe scripts/live_page_text.py track_record
(page name without a leading slash: Git Bash rewrites "/x" into a Windows path)
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SITES = Path(__file__).resolve().parent.parent.parent / "_website-gate" / "sites.json"
LOAD_TIMEOUT_MS = 120_000
SETTLE_MS = 7_000


def main() -> int:
    site = json.loads(SITES.read_text(encoding="utf-8"))["10q-equity-agent"]
    path = "/" + (sys.argv[1] if len(sys.argv) > 1 else "").lstrip("/")
    url = site["live_url"].rstrip("/") + site.get("live_prefix", "") + path
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)  # system Chrome, as webgate does
        page = browser.new_page()
        # networkidle, not ready_selector: found live, waiting on stMain alone hung on the
        # deployed app even though the page had fully rendered
        page.goto(url, wait_until="networkidle", timeout=LOAD_TIMEOUT_MS)
        page.wait_for_timeout(SETTLE_MS)
        sys.stdout.buffer.write(page.inner_text("body").encode("utf-8"))  # page has non-cp1252 chars (≈)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
