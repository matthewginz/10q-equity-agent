"""
Shared Gemini call helper, adapted from job-auto-applier/tailoring/gemini_client.py
(same free-tier fallback-chain pattern, proven live in that project).

Uses Gemini's free tier (ai.google.dev) rather than a paid API -- each model
has its own separate daily quota bucket, so generate() walks a priority
chain of models on RESOURCE_EXHAUSTED rather than failing the moment the
top choice is exhausted, and always re-tries the best model first on the
next call (no cached cooldown state) so it recovers the instant a quota
resets.
"""

import os
import threading
import time
from collections import Counter
from contextlib import contextmanager

from core import _net  # noqa: F401 -- must run before any network call, see its docstring
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

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
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:  # noqa: BLE001 -- no secrets.toml locally, or not running under streamlit
        return None


GEMINI_API_KEY = _load_gemini_api_key()

DEFAULT_MODEL = "gemini-3.5-flash"
MODEL_CHAIN = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
]

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


def _no_thinking(model: str) -> types.ThinkingConfig:
    """Found live: Gemma models reject thinking_budget (400) AND think by
    default -- with the pipeline's small output caps the hidden reasoning
    used the whole budget and the visible answer came back empty. Gemma
    takes thinking_level='minimal' instead; Gemini takes a zero budget."""
    if model.startswith("gemma"):
        return types.ThinkingConfig(thinking_level="minimal")
    return types.ThinkingConfig(thinking_budget=0)


def _call_model(model: str, system: str, user: str, max_output_tokens: int) -> str:
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
        except (ClientError, ServerError) as exc:
            retryable = isinstance(exc, ServerError) or getattr(exc, "status", None) == "RESOURCE_EXHAUSTED"
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


def generate(system: str, user: str, max_output_tokens: int = 2048) -> str:
    last_exc: Exception | None = None
    for candidate in getattr(_pin, "chain", None) or MODEL_CHAIN:
        try:
            text = _call_with_retries(candidate, system, user, max_output_tokens)
            MODEL_USAGE[candidate] += 1
            return text
        except ClientError as exc:
            # RESOURCE_EXHAUSTED = this model's free-tier quota is used up;
            # NOT_FOUND = this model name has been deprecated/retired.
            # INVALID_ARGUMENT is normally a real bad-request bug worth
            # surfacing immediately -- but found live: a real 400
            # INVALID_ARGUMENT on one specific (likely preview/experimental)
            # model in the chain did NOT reproduce seconds later against the
            # exact same prompt, so it's model-specific flakiness at least
            # some of the time, not a deterministic malformed request. Worth
            # one pass through the rest of the chain before giving up --
            # if every model rejects the same prompt, that's real signal;
            # if only one does, this recovers automatically.
            if getattr(exc, "status", None) in ("RESOURCE_EXHAUSTED", "NOT_FOUND", "INVALID_ARGUMENT"):
                last_exc = exc
                continue
            raise
        except EmptyResponse as exc:
            # a blank answer is a failed call, not a result -- try the next model
            last_exc = exc
            continue
        except ServerError as exc:
            # transient overload (503), not a quota issue -- worth trying
            # the next model rather than failing the whole analysis on one
            # model's momentary bad luck.
            last_exc = exc
            continue
    raise last_exc
