import sys
from pathlib import Path

import pytest
from google.genai.errors import ClientError, ServerError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import gemini_client as g

PER_DAY = ("You exceeded your current quota. Quota exceeded for metric: generate_content_free_tier_requests, "
           "limit: 20, model: m. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
PER_MINUTE = ("You exceeded your current quota. quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. "
              "Please retry in 12.5s.")


def _quota(message: str) -> ClientError:
    return ClientError(429, {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED"}})


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    g._COOLDOWN_UNTIL.clear()
    now = {"t": 1000.0}
    monkeypatch.setattr(g, "_now", lambda: now["t"])
    monkeypatch.setattr(g.time, "sleep", lambda s: now.__setitem__("t", now["t"] + s))
    monkeypatch.setattr(g, "MODEL_CHAIN", ["a", "b", "c"])
    return now


def test_day_quota_model_is_skipped_on_later_calls(monkeypatch):
    calls = []

    def fake(model, *_):
        calls.append(model)
        if model == "a":
            raise _quota(PER_DAY)
        return f"answer from {model}"

    monkeypatch.setattr(g, "_call_model", fake)
    assert g.generate("s", "u") == "answer from b"
    assert g.generate("s", "u") == "answer from b"
    assert calls == ["a", "b", "b"]          # "a" not retried within its cooldown


def test_day_quota_named_only_in_the_details_still_rests_an_hour():
    # the real shape: short message with a "retry in" hint, quota id only in details
    exc = ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "You exceeded your current quota. Please retry in 9.8s.",
        "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
    assert g._rest_seconds(exc) == g.DAY_COOLDOWN_S


def test_minute_quota_cools_down_only_for_the_stated_delay(monkeypatch, fresh):
    state = {"a_blocked": True}

    def fake(model, *_):
        if model == "a" and state["a_blocked"]:
            raise _quota(PER_MINUTE)
        return f"answer from {model}"

    monkeypatch.setattr(g, "_call_model", fake)
    assert g.generate("s", "u") == "answer from b"
    fresh["t"] += 13                          # past the 12.5s Google asked for
    state["a_blocked"] = False
    assert g.generate("s", "u") == "answer from a"


def test_waits_for_a_short_cooldown_instead_of_failing_when_every_model_is_busy(monkeypatch, fresh):
    def fake(model, *_):
        if fresh["t"] < 1010:                 # everything rate-limited for the first 10 seconds
            raise _quota(PER_MINUTE)
        return f"answer from {model}"

    monkeypatch.setattr(g, "_call_model", fake)
    assert g.generate("s", "u") == "answer from a"


def test_fails_cleanly_when_every_model_is_out_for_the_day(monkeypatch):
    monkeypatch.setattr(g, "_call_model", lambda model, *_: (_ for _ in ()).throw(_quota(PER_DAY)))
    with pytest.raises(g.AllModelsBusy):
        g.generate("s", "u")


def test_overload_moves_on_to_the_next_model(monkeypatch):
    def fake(model, *_):
        if model == "a":
            raise ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
        return f"answer from {model}"

    monkeypatch.setattr(g, "_call_model", fake)
    assert g.generate("s", "u") == "answer from b"


def test_dropped_connection_retries_the_same_model_without_resting_it(monkeypatch):
    import httpx
    calls = []

    def fake(model, *_):
        calls.append(model)
        if len(calls) == 1:
            raise httpx.ReadError("[WinError 10053] connection aborted")
        return f"answer from {model}"

    monkeypatch.setattr(g, "_call_model", fake)
    assert g.generate("s", "u") == "answer from a"
    assert calls == ["a", "a"] and "a" not in g._COOLDOWN_UNTIL


def test_rejected_thinking_setting_is_retried_with_the_other_one(monkeypatch):
    seen = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            seen.append(config.thinking_config)
            if config.thinking_config.thinking_budget == 0:
                raise ClientError(400, {"error": {"code": 400, "message": "invalid", "status": "INVALID_ARGUMENT"}})
            return type("R", (), {"text": "OK", "candidates": []})()

    monkeypatch.setattr(g, "_get_client", lambda: type("C", (), {"models": FakeModels()})())
    g._LEVEL_THINKING_MODELS.discard("new-lite")
    assert g._call_model("new-lite", "s", "u", 20) == "OK"
    assert "new-lite" in g._LEVEL_THINKING_MODELS             # remembered for next time
    assert g._call_model("new-lite", "s", "u", 20) == "OK"
    assert len(seen) == 3                                    # budget (rejected), level, level


def test_backtest_shares_only_high_quota_models_with_the_site(monkeypatch):
    monkeypatch.undo()                    # the real chain, not the fixture's fake one
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import backtest_accuracy as bt
    safe = bt.models_safe_on_shared_key(bt.MODELS)
    assert "gemma-4-26b-a4b-it" in safe
    assert set(safe) & set(g.MODEL_CHAIN) <= g.HIGH_QUOTA_MODELS
