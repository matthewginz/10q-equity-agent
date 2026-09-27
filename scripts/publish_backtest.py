"""Publish backtest results to the live site while a run is still going.

Run every few hours by the '10QAgent-PublishBacktest' scheduled task (and at the
end of each daily chunk via publish_backtest.bat). A full S&P 500 run takes days,
so waiting for it to finish meant the live Track Record page sat stale.

Steps, each one stopping the publish if it fails:
  0. analyst_comparison.csv is rebuilt (the page's numbers come from it)
  1. the data files parse (the backtest rewrites them in place, so a read can
     land mid-write; retry a few times before giving up)
  2. progress.json gets the fresh scored count + timestamp (the backtest only
     rewrites it at the start and end of a run)
  3. the website gate screens the site locally with the new data
  4. commit ONLY the data files and push (Streamlit Cloud deploys from main)
  5. wait until the live Track Record page shows the new timestamp, then run
     the gate against the live site

Exit code 0 = published and verified live (or nothing new); 1 = something failed,
details in logs/backtest.log.
"""
import csv
import io
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DIR = ROOT / "docs" / "backtest"
RESULTS_CSV = BACKTEST_DIR / "results.csv"
RAW_LOG = BACKTEST_DIR / "raw_runs.json"
ANALYST_CSV = BACKTEST_DIR / "analyst_comparison.csv"
PROGRESS_JSON = BACKTEST_DIR / "progress.json"
DATA_FILES = [ANALYST_CSV, PROGRESS_JSON, RAW_LOG, RESULTS_CSV]

GATE_DIR = ROOT.parent / "_website-gate"
GATE_PYTHON = GATE_DIR / "venv" / "Scripts" / "python.exe"
SITE = "10q-equity-agent"

VALID_TRIES = 6
VALID_WAIT_S = 10
DEPLOY_TIMEOUT_S = 15 * 60
DEPLOY_POLL_S = 60


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} publish: {msg}", flush=True)


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")


def rel(paths: list[Path]) -> list[str]:
    return [p.relative_to(ROOT).as_posix() for p in paths]


def csv_rows(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def data_problem(snapshot: dict[Path, bytes]) -> str | None:
    """None if every data file in the snapshot parses and looks whole, else what is wrong."""
    try:
        runs = json.loads(snapshot[RAW_LOG])
        rows = csv_rows(snapshot[RESULTS_CSV])
        if ANALYST_CSV in snapshot:
            csv_rows(snapshot[ANALYST_CSV])
    except (KeyError, ValueError, csv.Error) as exc:
        return f"{type(exc).__name__}: {exc}"
    if not rows or not runs:
        return "results.csv or raw_runs.json is empty"
    if any(None in r or None in r.values() for r in rows):
        return "results.csv has a ragged row (caught mid-write?)"
    return None


def take_valid_snapshot() -> dict[Path, bytes] | None:
    """Read the data files once and keep those exact bytes, so what gets committed
    is what was checked, even though the backtest keeps writing the files."""
    for attempt in range(1, VALID_TRIES + 1):
        snapshot = {p: p.read_bytes() for p in (RESULTS_CSV, RAW_LOG, ANALYST_CSV) if p.exists()}
        problem = data_problem(snapshot)
        if problem is None:
            return snapshot
        log(f"data not readable yet (try {attempt}/{VALID_TRIES}): {problem}")
        time.sleep(VALID_WAIT_S)
    return None


def refresh_comparison() -> bool:
    """Rebuild analyst_comparison.csv from the current results. The Track Record page's
    numbers come from this file, not results.csv, and the daily chain only rebuilds it
    once a chunk ends -- found live: a mid-run publish showed '2705 scored' over KPIs
    still counting 588."""
    python = ROOT / "venv" / "Scripts" / "python.exe"
    result = run([str(python), "-u", str(ROOT / "scripts" / "analyst_comparison.py")], timeout=3600)
    if result.returncode != 0:
        log(f"analyst_comparison.py failed (exit {result.returncode}): {result.stderr.strip()[-300:]}")
    return result.returncode == 0


def has_new_results() -> bool:
    return run(["git", "diff", "--quiet", "--", *rel([RESULTS_CSV, RAW_LOG, ANALYST_CSV])]).returncode != 0


def with_fresh_progress(snapshot: dict[Path, bytes]) -> tuple[dict[Path, bytes], int, str]:
    """Add progress.json with the snapshot's scored count and a new timestamp,
    and write it to disk too so the local gate renders it."""
    progress = json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    done = sum(1 for r in csv_rows(snapshot[RESULTS_CSV]) if r.get("models"))
    updated = datetime.now().isoformat(timespec="minutes")
    data = json.dumps({**progress, "filings_done": done, "updated": updated}, indent=1).encode("utf-8")
    tmp = PROGRESS_JSON.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    tmp.replace(PROGRESS_JSON)
    return {**snapshot, PROGRESS_JSON: data}, done, updated


def gate(mode_args: list[str]) -> bool:
    result = run([str(GATE_PYTHON), str(GATE_DIR / "webgate.py"), SITE, *mode_args], timeout=1800)
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-8:])
    log(f"website gate {' '.join(mode_args) or '(quick)'} -> exit {result.returncode}\n{tail}")
    return result.returncode == 0


def stage_snapshot(snapshot: dict[Path, bytes]) -> bool:
    """Put the checked bytes straight into git's index (not whatever is on disk now)."""
    for path, data in snapshot.items():
        # --path applies the same line-ending filters `git add` would
        blob = subprocess.run(["git", "hash-object", "-w", "--stdin", "--path", rel([path])[0]],
                              cwd=ROOT, input=data, capture_output=True)
        if blob.returncode != 0:
            log(f"hash-object failed for {path.name}: {blob.stderr.decode(errors='replace').strip()}")
            return False
        sha = blob.stdout.decode().strip()
        staged = run(["git", "update-index", "--cacheinfo", f"100644,{sha},{rel([path])[0]}"])
        if staged.returncode != 0:
            log(f"update-index failed for {path.name}: {staged.stderr.strip()}")
            return False
    return True


def commit_and_push(snapshot: dict[Path, bytes], done: int) -> bool:
    other_staged = run(["git", "diff", "--cached", "--name-only"]).stdout.split()
    if set(other_staged) - set(rel(DATA_FILES)):
        log(f"other files are staged ({other_staged}); not committing them with data -- skipping this publish")
        return False
    if not stage_snapshot(snapshot):
        return False
    commit = run(["git", "commit", "-m", f"data: backtest snapshot, {done} filings scored (auto)"])
    if commit.returncode != 0:
        log(f"commit failed: {commit.stdout.strip()} {commit.stderr.strip()}")
        return False
    push = run(["git", "push", "origin", "main"])
    if push.returncode != 0:
        log(f"push failed (commit kept locally, goes up with the next push): {push.stderr.strip()}")
        return False
    return True


def live_shows(updated: str) -> bool:
    """Poll the deployed Track Record page until it shows this publish's timestamp."""
    stamp = f"Last update {updated.replace('T', ' ')}"
    deadline = time.monotonic() + DEPLOY_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(DEPLOY_POLL_S)
        check = run([str(GATE_PYTHON), str(ROOT / "scripts" / "live_page_text.py"), "track_record"], timeout=180)
        if stamp in check.stdout:
            return True
        log(f"live page not updated yet ({check.stderr.strip()[-160:] or 'stamp not found'})")
    return False


def main() -> int:
    if not has_new_results():
        log("no new backtest results, nothing to publish")
        return 0
    if not refresh_comparison():
        log("not publishing: the page's numbers would disagree with the new results")
        return 1
    snapshot = take_valid_snapshot()
    if snapshot is None:
        log("data files never became readable, not publishing")
        return 1
    snapshot, done, updated = with_fresh_progress(snapshot)
    log(f"{done} filings scored, stamp {updated}")
    if not gate([]):
        log("local gate failed, not pushing -- see _website-gate/reports/")
        return 1
    if not commit_and_push(snapshot, done):
        return 1
    log("pushed; waiting for Streamlit Cloud to redeploy")
    if not live_shows(updated):
        log(f"live site still not showing the new data after {DEPLOY_TIMEOUT_S // 60} min")
        return 1
    if not gate(["--live"]):
        log("LIVE site failed the gate after this publish -- check _website-gate/reports/")
        return 1
    log(f"published {done} filings and verified the live site")
    return 0


if __name__ == "__main__":
    sys.exit(main())
