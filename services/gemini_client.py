"""
Shared Gemini call helper, adapted from job-auto-applier/tailoring/gemini_client.py
(same free-tier fallback-chain pattern, proven live in that project).

Uses Gemini's free tier (ai.google.dev) rather than a paid API -- each model
has its own separate daily quota bucket, so generate() walks a priority
chain of models on RESOURCE_EXHAUSTED rather than failing the moment the
top choice is exhausted, and always re-tries the best model first on the
next call.

Found live 2026-09-25 ("failed on every fallback model" for real users): only
5 of the 11 chained models still answered on the free key -- three were
retired (404), two rejected thinking_budget=0 and one was an alias of a model
the backtest was using -- so the whole site had ~100 requests/day, about 7
comparisons. Now: dead models dropped, the thinking setting adapts per model,
Gemma (a far larger free quota) backs the chain, and a model that hits a
limit rests for as long as that limit lasts instead of being re-hit on every
call. If every model is briefly rate-limited, generate() waits for the first
to free up rather than failing.
"""

import os
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager

from core import _net  # noqa: F401 -- must run before any network call, see its docstring
from dotenv import load_dotenv
from google import genai
from google.genai import types
import httpx
from google.genai.errors import ClientError, ServerError

# Found live: a dropped connection mid-call (httpx ReadError, WinError 10053)
# killed a whole analysis -- network blips get the same treatment as a 503.
_TRANSIENT = (ServerError, httpx.TransportError)

load_dotenv()


def _load_gemini_api_key() -> str | None:
    """os.getenv() covers local dev (.env via load_dotenv above). On
    Streamlit Community Cloud, secrets set in the dashboard are documented
    to mirror into os.environ automatically -- but only lazily, the first
    time anything actually touches st.secrets. This module never did, so
    that mirroring never fired and os.getenv came back empty even with a
    correctly-configured secret (confirmed live: the secret was
    byte-exact correct and the key itself verified valid against Google's
    API directly, yet the app still reported "not set"). Reading
    st.secrets directly, guarded for the plain-Python/no-secrets-file
    case (local dev without .streamlit/secrets.toml raises here), fixes
    it regardless of whether anything else already triggered the mirror."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:  # noqa: BLE001 -- no secrets.toml locally, or not running under streamlit
            return None
    return _clean_key(key)


def _clean_key(key: str | None) -> str | None:
    """Found live: a key pasted into the Streamlit secrets box can carry
    stray whitespace or an extra layer of quotes, and Gemini answers every
    model with a bare 400 INVALID_ARGUMENT instead of "bad key"."""
    if key is None:
        return None
    cleaned = key.strip().strip("\"'").strip()
    return cleaned or None


GEMINI_API_KEY = _load_gemini_api_key()

DEFAULT_MODEL = "gemini-3.5-flash"
# Probed live on the app's key 2026-09-25. Removed: gemini-2.5-flash / -pro /
# -flash-lite (retired, 404) and gemini-flash-latest (an alias of
# gemini-3.8-flash, now named directly so the backtest can't share it).
MODEL_CHAIN = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    # Last resort, with a far larger free quota (the backtest has made thousands
    # of calls a day on it): slower and a bit less polished, never a failure.
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
]
# Models big enough to share between the site and the nightly backtest; every
# other model in MODEL_CHAIN is reserved for the site (see backtest_accuracy.py).
HIGH_QUOTA_MODELS = frozenset({"gemma-4-26b-a4b-it"})

_client: genai.Client | None = None

# Successful calls per model, process-wide. The backtest diffs this around each
# filing to record which model(s) actually answered -- the fallback chain means
# a big run can silently mix models once the top one's quota runs out.
MODEL_USAGE: Counter[str] = Counter()


def use_api_key(key: str) -> None:
    """Point every later call at a different key -- the backtest uses its own
    key (a separate Google project) so it never spends the public app's
    free-tier quota."""
    global GEMINI_API_KEY, _client
    GEMINI_API_KEY = key
    _client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY not set -- copy .env.example to .env and add a free key "
                "from https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


class EmptyResponse(RuntimeError):
    """The model returned no visible text -- never a usable answer."""


# Models that reject thinking_budget=0 (400 INVALID_ARGUMENT) and need
# thinking_level="minimal" instead. Seeded from the 2026-09-25 probe; any other
# model that rejects the budget is added the first time it does.
_LEVEL_THINKING_MODELS: set[str] = {"gemini-3.5-flash-lite", "gemini-flash-lite-latest"}


def _no_thinking(model: str) -> types.ThinkingConfig:
    """Found live: Gemma models reject thinking_budget (400) AND think by
    default -- with the pipeline's small output caps the hidden reasoning
    used the whole budget and the visible answer came back empty. Gemma and
    the newer lite models take thinking_level='minimal'; the rest take a
    zero budget."""
    if model.startswith("gemma") or model in _LEVEL_THINKING_MODELS:
        return types.ThinkingConfig(thinking_level="minimal")
    return types.ThinkingConfig(thinking_budget=0)


def _call_model(model: str, system: str, user: str, max_output_tokens: int) -> str:
    try:
        return _call_model_once(model, system, user, max_output_tokens)
    except ClientError as exc:
        uses_budget = not (model.startswith("gemma") or model in _LEVEL_THINKING_MODELS)
        if getattr(exc, "status", None) != "INVALID_ARGUMENT" or not uses_budget:
            raise
        _LEVEL_THINKING_MODELS.add(model)       # this model wants the other setting; remember it
        return _call_model_once(model, system, user, max_output_tokens)


def _call_model_once(model: str, system: str, user: str, max_output_tokens: int) -> str:
    thinking = _no_thinking(model)
    response = _get_client().models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_output_tokens,
            thinking_config=thinking,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        reason = response.candidates[0].finish_reason if response.candidates else "no candidates"
        raise EmptyResponse(f"{model} returned no text (finish_reason={reason})")
    return text


# Waits before re-trying the SAME model on a transient 503 overload or a
# per-minute 429. Empty for the live app (fall straight through to the next
# model, fastest for an interactive user). The backtest pins a single model
# for consistency and sets this, so a momentary overload waits instead of
# either failing the case or silently switching models.
SAME_MODEL_RETRY_WAITS: tuple[float, ...] = ()


def _call_with_retries(model: str, system: str, user: str, max_output_tokens: int) -> str:
    for wait in (*SAME_MODEL_RETRY_WAITS, None):
        try:
            return _call_model(model, system, user, max_output_tokens)
        except (ClientError, *_TRANSIENT) as exc:
            retryable = isinstance(exc, _TRANSIENT) or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"
            if wait is None or not retryable:
                raise
            time.sleep(wait)
    raise AssertionError("unreachable")


_pin = threading.local()


@contextmanager
def pinned_model(model: str):
    """Within this block, on this thread only, generate() uses exactly
    `model` -- no fallback. Lets the backtest run one worker thread per
    model with every step of a filing answered by the same model."""
    previous = getattr(_pin, "chain", None)
    _pin.chain = [model]
    try:
        yield
    finally:
        _pin.chain = previous


# ── Rest periods ───────────────────────────────────────────────────────────
DAY_COOLDOWN_S = 3600          # daily quota used up: re-check hourly (it resets at midnight Pacific)
MINUTE_COOLDOWN_S = 60         # per-minute limit when Google doesn't say how long
OVERLOAD_COOLDOWN_S = 30       # 503 / network blip
FLAKY_COOLDOWN_S = 20          # empty answer or a one-off INVALID_ARGUMENT
RETIRED_COOLDOWN_S = 24 * 3600
# Per-minute limits always lift within a minute, so waiting this long turns
# "every model is busy" into a short wait instead of a failed analysis
# (stress test 2026-09-25: at 75s, 3 of 6 Gemma-only analyses gave up).
MAX_WAIT_FOR_A_FREE_MODEL_S = 120
BLIP_RETRY_WAIT_S = 1.0
_COOLDOWN_UNTIL: dict[str, float] = {}


class AllModelsBusy(RuntimeError):
    """Every model in the chain is resting for longer than it's worth waiting."""


def _now() -> float:
    return time.monotonic()


def _rest_seconds(exc: Exception) -> float:
    """How long a model that just failed with `exc` should be left alone."""
    if isinstance(exc, EmptyResponse):
        return FLAKY_COOLDOWN_S
    if isinstance(exc, _TRANSIENT):
        return OVERLOAD_COOLDOWN_S
    # Found live: the quota id ("...PerDay...") is only in the error's details, not its
    # message, and a daily limit still says "retry in 10s" -- so read the whole error.
    status, message = getattr(exc, "status", None), f"{getattr(exc, 'message', '')} {exc}"
    if status == "NOT_FOUND":
        return RETIRED_COOLDOWN_S
    if status == "RESOURCE_EXHAUSTED":
        if "PerDay" in message:
            return DAY_COOLDOWN_S
        delay = re.search(r"retry in ([\d.]+)\s*s", message, re.I) or re.search(r"retryDelay\W+(\d+)", message)
        return float(delay.group(1)) if delay else MINUTE_COOLDOWN_S
    return FLAKY_COOLDOWN_S


def _is_fallthrough(exc: Exception) -> bool:
    """Failures that mean "try another model", not "this request is broken"."""
    if isinstance(exc, (EmptyResponse, *_TRANSIENT)):
        return True
    return getattr(exc, "status", None) in ("RESOURCE_EXHAUSTED", "NOT_FOUND", "INVALID_ARGUMENT")


def _generate_pinned(chain: list[str], system: str, user: str, max_output_tokens: int) -> str:
    """The backtest's path: exactly the pinned model, riding out limits with
    SAME_MODEL_RETRY_WAITS, never switching models mid-filing."""
    last_exc: Exception | None = None
    for candidate in chain:
        try:
            text = _call_with_retries(candidate, system, user, max_output_tokens)
            MODEL_USAGE[candidate] += 1
            return text
        except (ClientError, EmptyResponse, *_TRANSIENT) as exc:
            if not _is_fallthrough(exc):
                raise
            last_exc = exc
    raise last_exc


def _call_model_riding_out_blips(model: str, system: str, user: str, max_output_tokens: int) -> str:
    """Found in the stress test: a dropped connection (WinError 10053) is this
    machine's network, not the model -- resting the model for it left nothing
    to fall back to. Retry the same model once first."""
    try:
        return _call_model(model, system, user, max_output_tokens)
    except httpx.TransportError:
        time.sleep(BLIP_RETRY_WAIT_S)
        return _call_model(model, system, user, max_output_tokens)


def generate(system: str, user: str, max_output_tokens: int = 2048) -> str:
    pinned = getattr(_pin, "chain", None)
    if pinned:
        return _generate_pinned(pinned, system, user, max_output_tokens)
    give_up_at = _now() + MAX_WAIT_FOR_A_FREE_MODEL_S
    last_exc: Exception | None = None
    while True:
        for candidate in [m for m in MODEL_CHAIN if _COOLDOWN_UNTIL.get(m, 0.0) <= _now()]:
            try:
                text = _call_model_riding_out_blips(candidate, system, user, max_output_tokens)
                MODEL_USAGE[candidate] += 1
                return text
            except (ClientError, EmptyResponse, *_TRANSIENT) as exc:
                if not _is_fallthrough(exc):
                    raise
                _COOLDOWN_UNTIL[candidate] = _now() + _rest_seconds(exc)
                last_exc = exc
        next_free = min(_COOLDOWN_UNTIL.get(m, 0.0) for m in MODEL_CHAIN)
        if next_free > give_up_at:
            raise AllModelsBusy(
                "Every free AI model this site uses has hit its limit for now (Google's free tier resets "
                "daily at midnight Pacific). Please try again later."
            ) from last_exc
        time.sleep(max(next_free - _now(), 0.5))     # a per-minute limit is about to lift: wait, don't fail
