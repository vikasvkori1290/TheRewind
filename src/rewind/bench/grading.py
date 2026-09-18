"""Answer grading: every expected group must appear in the answer."""

from __future__ import annotations

import re

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_SPACE = re.compile(r"\s+")

# Phrases that show the model admitted it has no record, for control questions.
ADMITS_UNKNOWN = (
    "not ", "no record", "don't", "do not", "didn't", "did not", "haven't",
    "have not", "never", "no information", "wasn't", "was not", "unable",
    "can't find", "cannot find", "no mention", "not_found",
)


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‑", "-")
    text = _THOUSANDS.sub("", text)
    return _SPACE.sub(" ", text)


def grade(answer: str, expect: list[list[str]]) -> bool:
    """True when, for every group, at least one alternative occurs in the answer."""
    norm = normalize(answer)
    return all(any(normalize(alt) in norm for alt in group) for group in expect)


def grade_control(answer: str) -> bool:
    """A control question asks about something never discussed; pass if the model says so."""
    norm = normalize(answer)
    return any(marker in norm for marker in ADMITS_UNKNOWN)
