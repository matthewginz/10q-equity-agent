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


def test_done_and_error_states_fill_the_bar():
    assert progress_of(_state(done=True, finished_at=84.0), now=90.0) == (1.0, "Done in 1m 25s")
    frac, label = progress_of(_state(error="boom", done=True, finished_at=1.0), now=2.0)
    assert frac == 1.0 and "Stopped" in label
