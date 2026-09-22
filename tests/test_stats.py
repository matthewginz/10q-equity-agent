import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.stats import binomial_p_value, hit_rate, wilson_interval


def test_four_of_five_is_not_significant():
    # the original 5-call "contrarian" result: a coin flip does this ~19% of the time
    assert binomial_p_value(4, 5) == pytest.approx(6 / 32)


def test_p_value_edges():
    assert binomial_p_value(0, 10) == pytest.approx(1.0)
    assert binomial_p_value(10, 10) == pytest.approx(0.5 ** 10)
    assert binomial_p_value(0, 0) is None


def test_wilson_matches_known_value():
    low, high = wilson_interval(8, 12)
    assert low == pytest.approx(0.3906, abs=1e-3)
    assert high == pytest.approx(0.8619, abs=1e-3)


def test_wilson_stays_inside_zero_one():
    low, high = wilson_interval(5, 5)
    assert 0.0 <= low <= high <= 1.0
    assert high == pytest.approx(1.0)


def test_hit_rate_bundles_everything():
    result = hit_rate(60, 100)
    assert result.rate == pytest.approx(0.6)
    assert result.ci_low < 0.6 < result.ci_high
    assert result.p_value == pytest.approx(0.0284, abs=1e-3)
    empty = hit_rate(0, 0)
    assert empty.rate is None and empty.ci_low is None and empty.p_value is None
