"""Compaction strategies.

Every strategy implements `Compactor`, so the session, the demo and the
benchmark can swap the baseline for Rewind with a configuration change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rewind.llm import LLM
from rewind.messages import find_cut, prepend_text, response_text, to_text
from rewind.metrics import Usage

SUMMARY_PROMPT = (
    "Summarize the conversation below in under 200 words. Keep decisions, "
    "names, numbers and open tasks. Do not copy code.\n\n"
)


@dataclass
class CompactionResult:
    messages: list[dict]
    removed_messages: int
    usage: Usage = field(default_factory=Usage)


class Compactor(Protocol):
    name: str

    def compact(self, messages: list[dict], session_id: str) -> CompactionResult | None:
        """Return shortened history, or None when nothing can be compacted."""
        ...


class PlainCompactor:
    """Baseline: replace old turns with a summary. Details in them are lost."""

    name = "plain"

    def __init__(self, llm: LLM, keep_recent: int, summary_max_tokens: int):
        self._llm = llm
        self._keep_recent = keep_recent
        self._summary_max_tokens = summary_max_tokens

    def compact(self, messages: list[dict], session_id: str) -> CompactionResult | None:
        cut = find_cut(messages, self._keep_recent)
        if cut == 0:
            return None
        old, kept = messages[:cut], messages[cut:]
        summary, usage = self._summarize(old)
        header = f"[Earlier conversation, compacted]\n{summary}"
        return CompactionResult(
            messages=[prepend_text(kept[0], header), *kept[1:]],
            removed_messages=len(old),
            usage=usage,
        )

    def _summarize(self, old: list[dict]) -> tuple[str, Usage]:
        transcript = "\n\n".join(to_text(m) for m in old)
        response = self._llm.create(
            system="You write concise, faithful conversation summaries.",
            messages=[{"role": "user", "content": SUMMARY_PROMPT + transcript}],
            max_tokens=self._summary_max_tokens,
        )
        usage = Usage()
        usage.add_response(response)
        return response_text(response), usage
