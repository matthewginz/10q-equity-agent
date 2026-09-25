import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.live_run import DATA_SHARE, RunState, progress_of


def _state(**kwargs) -> RunState:
    state = RunState("AAPL")
    state.started = 0.0
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def test_data_stage_shows_typical_estimate():
    frac, label = progress_of(_state(stage=2), now=3.0)
    assert 0 < frac < DATA_SHARE
    assert "MD&A" in label and "usually about" in label


def test_estimate_uses_measured_pace_after_first_steps():
    # 2 steps took 20s each -> 4 left at 20s, 5s into the current one -> ~75s
    state = _state(pipeline_started=10.0, last_step_at=50.0, current_title="Segment detail",
                   steps=[("1", "a"), ("2", "b")])
    frac, label = progress_of(state, now=55.0)
    assert frac == DATA_SHARE + (1 - DATA_SHARE) * 2 / 6
    assert "Step 3/6: Segment detail" in label
    assert "1m 15s left" in label


def test_estimate_never_goes_negative_or_zero():
    state = _state(pipeline_started=0.0, last_step_at=5.0, steps=[("1", "a")] * 5, current_title="Final")
    _, label = progress_of(state, now=500.0)
    assert "about 5s left" in label


def test_same_filing_same_day_is_analyzed_once(monkeypatch):
    from types import SimpleNamespace

    import ui.live_run as lr
    from services.pipeline import AnalysisRun, StepResult

    filing = SimpleNamespace(company_name="Apple Inc.", report_date="2026-06-27", accession_number="0000320193-26-1")
    monkeypatch.setattr(lr, "_fetch_inputs", lambda state: (filing, {}, {}, "text", None))
    calls = []

    def fake_analysis(*args, on_step, on_step_done, **kwargs):
        calls.append(1)
        for i, title in enumerate(["1. A", "2. B"], start=1):
            on_step(i, title)
            on_step_done(i, title, f"out {i}")
        on_step_done(3, "Final Equity Stance", "### Final Equity Research Stance: Bullish")
        return AnalysisRun("Apple Inc.", "AAPL", "2026-06-27", [StepResult("1. A", "out 1"), StepResult("2. B", "out 2")],
                           "### Final Equity Research Stance: Bullish")

    monkeypatch.setattr(lr, "run_analysis", fake_analysis)
    lr._ANALYSIS_CACHE.clear()
    first, second = lr.RunState("AAPL"), lr.RunState("AAPL")
    lr._worker(first)
    lr._worker(second)
    assert len(calls) == 1                                   # the second run cost no AI requests
    assert second.bundle is not None and second.error is None
    assert second.steps == first.steps == [("1. A", "out 1"), ("2. B", "out 2"),
                                           ("Final Equity Stance", "### Final Equity Research Stance: Bullish")]


def test_done_and_error_states_fill_the_bar():
    assert progress_of(_state(done=True, finished_at=84.0), now=90.0) == (1.0, "Done in 1m 25s")
    frac, label = progress_of(_state(error="boom", done=True, finished_at=1.0), now=2.0)
    assert frac == 1.0 and "Stopped" in label
