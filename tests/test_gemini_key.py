import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.gemini_client import _clean_key


def test_clean_key_strips_paste_noise():
    assert _clean_key("  AQ.abc123  ") == "AQ.abc123"
    assert _clean_key('"AQ.abc123"') == "AQ.abc123"
    assert _clean_key("'AQ.abc123'\n") == "AQ.abc123"


def test_clean_key_blank_is_none():
    assert _clean_key("   ") is None
    assert _clean_key(None) is None
