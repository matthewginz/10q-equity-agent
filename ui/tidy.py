"""
Display-only cleanup of the pipeline's text, so the page doesn't repeat what
it already shows. The model output itself is never changed -- the backtest
and the stance badge read the raw text -- this only trims what's rendered.

Found in 759 real runs (docs/backtest/raw_runs.json):
- the final step opens with "### Final Equity Research Stance: Bullish",
  right under a BULLISH badge;
- steps open with a title block restating the step and the filing's date
  and period ("**Equity Research Snapshot: NXP (NXPI)** / **Date:** ..."),
  which the company card already shows;
- steps open with a content-free lead-in ("Based on the provided 10-Q
  filing for the period ended March 29, 2026, here is the extracted
  information:"), repeated per step and, in compare mode, per company;
- the final step's section titles are long and echo their own body ("The
  single biggest supporting factor and the single biggest risk to the
  thesis" followed by "The single biggest supporting factor is ...").

Conservative on purpose: an opening is only dropped when it matches one of
these shapes, so an opening that carries a finding ("there is insufficient
information to identify risks") always stays.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,6}\s.*|\*\*[^*]+\*\*)$")
_TITLE_WORDS = re.compile(r"snapshot|equity research|stance|cross-check|consistency check", re.I)
_METADATA = re.compile(r"^\*\*(date[^*]*|period[^*]*|reporting period[^*]*|company|ticker)\s*:?\s*\*\*", re.I)
_LEAD_IN_START = re.compile(r"^(based on|upon|after reviewing|as an equity research analyst|below is|here is|here are)\b", re.I)
# \b matters: found in testing, a bare "here is" matched inside "tHERE IS insufficient information".
_LEAD_IN_MARKER = re.compile(
    r"\b(the following|here is|here are|below is|as follows|the requested|the extracted|are ranked below"
    r"|I have performed a cross-check)\b", re.I)
# A process-only first sentence; the rest of its paragraph can carry a finding, so only the sentence goes.
_PROCESS_SENTENCE = re.compile(
    r"^As an equity research analyst, I have (?:cross-referenced|performed a cross-check|reviewed|compared)"
    r"[^.]{0,300}?\.\s*", re.I)
MAX_LEAD_IN_CHARS = 400

_SECTION_TITLES = [   # final-step section headings -> short form
    (re.compile(r"^(market )?sentiment$", re.I), "Sentiment"),
    (re.compile(r"^why it'?s at this price$", re.I), "Why it's at this price"),
    (re.compile(r"^(the )?single biggest supporting factor.*|^supporting factor and risk$", re.I),
     "Biggest support and biggest risk"),
    (re.compile(r"^(analyst )?recommendation$", re.I), "Recommendation"),
]
_BOLD_LINE = re.compile(r"^\*\*(?:\d+\.\s*)?(.+?)\*\*$")


def _first_paragraph(lines: list[str]) -> list[str]:
    end = next((i for i, line in enumerate(lines) if not line.strip()), len(lines))
    return lines[:end]


def _drop_repeated_opening(lines: list[str]) -> list[str]:
    while lines:
        while lines and not lines[0].strip():
            lines = lines[1:]
        if not lines:
            break
        first = lines[0].strip()
        if (_HEADING.match(first) and _TITLE_WORDS.search(first)) or _METADATA.match(first):
            lines = lines[1:]
            continue
        trimmed = _PROCESS_SENTENCE.sub("", first, count=1)
        if trimmed != first:
            lines = ([trimmed] if trimmed else []) + lines[1:]
            continue
        paragraph = " ".join(line.strip() for line in _first_paragraph(lines))
        is_lead_in = _LEAD_IN_START.match(paragraph) and len(paragraph) <= MAX_LEAD_IN_CHARS and (
            (_LEAD_IN_MARKER.search(paragraph) and paragraph.endswith((":", ".")))
            or paragraph.endswith(":"))   # "Based on the ... filing for the period ended March 31, 2026:"
        if is_lead_in:
            lines = lines[len(_first_paragraph(lines)):]
            continue
        break
    return lines


def _short_section_title(line: str) -> str:
    match = _BOLD_LINE.match(line.strip())
    if not match:
        return line
    for pattern, short in _SECTION_TITLES:
        if pattern.match(match.group(1).strip()):
            return f"**{short}**"
    return line


def tidy_output(text: str) -> str:
    """The pipeline text with its repeated opening removed and the final
    step's section titles shortened. Never returns an empty string."""
    kept = _drop_repeated_opening(text.strip().splitlines())
    if not any(line.strip() for line in kept):
        return text
    return "\n".join(_short_section_title(line) for line in kept).strip()
