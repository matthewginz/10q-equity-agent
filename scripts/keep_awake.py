"""Keep the public Streamlit app from sleeping, and prove it actually renders.

Streamlit Community Cloud sleeps apps after 12h with no real viewer traffic, and
a plain HTTP ping doesn't count. So this opens each URL in headless Chromium as a
logged-out visitor, clicks "Yes, get this app back up!" if the sleep page shows,
and waits for text only the rendered app shows (the tab title alone is set before
anything renders, so an error page used to pass). Fails on Streamlit's
share.streamlit.io/errors/not_found page or an in-app error.

Run by the Scheduled Task "10Q-Streamlit-KeepAwake" (every 2h, this PC) and by
.github/workflows/keep-awake.yml (backup when the PC is off; a failure emails the
repo owner).
"""
import sys
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

# url -> text that appears only once the real app has rendered
CHECKS = {
    "https://10q-equity-agent.streamlit.app/": "Run analysis",
    # old deployment still linked from the Trading/QA resume PDFs (moved/app.py)
    "https://10q-equity-agent-lu7wjqswdyw7catzwyzq9j.streamlit.app/": "This app has moved",
}
BAD_TEXT = ("You do not have access", "does not exist", "Page not found", "Oh no", "Error running app")
RENDER_TIMEOUT_S = 180
LOG = Path(__file__).resolve().parent.parent / "logs" / "keep_awake.log"


def verdict(url: str, text: str, expect: str) -> str | None:
    """'ok' once the app rendered, 'FAILED ...' on an error page, None while still loading."""
    if "errors/not_found" in url:
        return "FAILED redirected to share.streamlit.io/errors/not_found"
    if expect.lower() in text.lower():
        return "ok"
    bad = next((b for b in BAD_TEXT if b.lower() in text.lower()), None)
    return f"FAILED page shows {bad!r}" if bad else None


def visible_text(page) -> str:
    # the app renders inside an iframe on *.streamlit.app, so read every frame
    return " ".join(f.evaluate("document.body ? document.body.innerText : ''") for f in page.frames)


def visit(page, url: str, expect: str) -> str:
    page.goto(url, timeout=60_000)
    page.wait_for_timeout(8_000)
    woke = False
    button = page.get_by_role("button", name="get this app back up", exact=False)
    if button.count():
        button.first.click()
        woke = True
    for _ in range(RENDER_TIMEOUT_S // 3):
        result = verdict(page.url, visible_text(page), expect)
        if result:
            return f"{result} (woken)" if woke and result == "ok" else result
        page.wait_for_timeout(3_000)
    return f"FAILED {expect!r} never appeared in {RENDER_TIMEOUT_S}s (url {page.url})"


def main() -> int:
    LOG.parent.mkdir(exist_ok=True)
    failed = False
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for url, expect in CHECKS.items():
            context = browser.new_context()  # fresh cookies: a logged-out visitor
            page = context.new_page()
            try:
                status = visit(page, url, expect)
            except Exception as e:  # logged and reflected in the exit code
                status = f"FAILED {type(e).__name__}: {e}".splitlines()[0]
            finally:
                context.close()
            failed |= status.startswith("FAILED")
            line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {url} {status}"
            print(line, flush=True)
            with LOG.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        browser.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
