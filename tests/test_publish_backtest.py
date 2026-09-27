import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import publish_backtest as pub

HEADER = b"ticker,filing_date,signal,models\n"


def _snapshot(results: bytes, runs: bytes = b'[{"ticker": "AAA"}]', analyst: bytes = HEADER) -> dict:
    return {pub.RESULTS_CSV: results, pub.RAW_LOG: runs, pub.ANALYST_CSV: analyst}


def test_whole_files_pass():
    assert pub.data_problem(_snapshot(HEADER + b"AAA,2024-08-01,Bullish,gemma\n")) is None


def test_row_cut_off_mid_write_is_caught():
    assert "ragged" in pub.data_problem(_snapshot(HEADER + b"AAA,2024-08-01,Bullish,gemma\nBBB,2024-08\n"))


def test_half_written_json_is_caught():
    assert "JSONDecodeError" in pub.data_problem(_snapshot(HEADER + b"AAA,2024-08-01,Bullish,gemma\n", runs=b'[{"tick'))


def test_empty_results_are_not_published():
    assert "empty" in pub.data_problem(_snapshot(HEADER))


def test_missing_analyst_file_is_allowed():
    snap = _snapshot(HEADER + b"AAA,2024-08-01,Bullish,gemma\n")
    del snap[pub.ANALYST_CSV]
    assert pub.data_problem(snap) is None


def test_fresh_progress_counts_scored_rows_and_keeps_other_fields(tmp_path, monkeypatch):
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"filings_in_window": 2987, "filings_done": 54, "updated": "2026-09-24T05:11"}))
    monkeypatch.setattr(pub, "PROGRESS_JSON", progress)
    snap = _snapshot(HEADER + b"AAA,2024-08-01,Bullish,gemma\nBBB,2024-08-02,Bearish,\n")

    new_snap, done, updated = pub.with_fresh_progress(snap)

    written = json.loads(progress.read_text())
    assert done == 1  # BBB has no model yet, so it isn't scored
    assert written == json.loads(new_snap[progress])
    assert written["filings_in_window"] == 2987 and written["filings_done"] == 1 and written["updated"] == updated
