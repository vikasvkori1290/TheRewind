"""Compaction strategies.

Every strategy implements `Compactor`, so the session, the demo and the
benchmark can swap the baseline for Rewind with a configuration change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rewind.llm import LLM
from rewind.messages import (
    HEADER_PREFIX,
    classify,
    find_cut,
    make_gist,
    prepend_text,
    response_text,
    split_turns,
    strip_header,
    to_text,
)
from rewind.metrics import Usage
from rewind.store import ArchiveStore

SUMMARY_PROMPT = (
    "Summarize the conversation below in under 200 words. Keep decisions, "
    "names, numbers and open tasks. Do not copy code.\n\n"
)


@dataclass
class CompactionResult:
    messages: list[dict]
    removed_messages: int
    usage: Usage = field(default_factory=Usage)
    archived_ids: list[str] = field(default_factory=list)


class Compactor(Protocol):
    name: str

    def compact(self, messages: list[dict], session_id: str) -> CompactionResult | None:
        """Return shortened history, or None when nothing can be compacted."""
        ...


class NullCompactor:
    """Never compacts: the upper bound for accuracy (and cost) in benchmarks."""

    name = "none"

    def compact(self, messages, session_id):
        return None


class _Summarizer:
    def __init__(self, llm: LLM, max_tokens: int):
        self._llm = llm
        self._max_tokens = max_tokens

    def __call__(self, old: list[dict]) -> tuple[str, Usage]:
        # The previous header stays in the transcript, so the summary is a
        # running summary of everything compacted so far.
        transcript = "\n\n".join(to_text(m) for m in old)
        response = self._llm.create(
            system="You write concise, faithful conversation summaries.",
            messages=[{"role": "user", "content": SUMMARY_PROMPT + transcript}],
            max_tokens=self._max_tokens,
        )
        usage = Usage()
        usage.add_response(response)
        return response_text(response), usage


class PlainCompactor:
    """Baseline: replace old turns with a summary. Details in them are lost."""

    name = "plain"

    def __init__(self, llm: LLM, keep_recent: int, summary_max_tokens: int):
        self._summarize = _Summarizer(llm, summary_max_tokens)
        self._keep_recent = max(1, keep_recent)

    def compact(self, messages, session_id):
        cut = find_cut(messages, self._keep_recent)
        if cut == 0:
            return None
        old, kept = messages[:cut], messages[cut:]
        summary, usage = self._summarize(old)
        header = f"{HEADER_PREFIX}\n{summary}"
        return CompactionResult(
            messages=[prepend_text(kept[0], header), *kept[1:]],
            removed_messages=len(old),
            usage=usage,
        )


class RewindCompactor:
    """Archive old turns verbatim, then summarize them and leave pointers.

    The header lists `§id → gist` for the most recent archived turns so the
    model can recall exact text by ID; older ones stay reachable by search.
    """

    name = "rewind"

    def __init__(self, llm: LLM, archive: ArchiveStore, keep_recent: int,
                 summary_max_tokens: int, max_pointers: int = 40):
        self._summarize = _Summarizer(llm, summary_max_tokens)
        self._archive = archive
        self._keep_recent = max(1, keep_recent)
        self._max_pointers = max_pointers
        self._archived_turns = 0

    def compact(self, messages, session_id):
        cut = find_cut(messages, self._keep_recent)
        if cut == 0:
            return None
        old, kept = messages[:cut], messages[cut:]

        archived = []
        for turn in split_turns(old):
            text = "\n\n".join(to_text(strip_header(m)) for m in turn)
            self._archived_turns += 1
            rec = self._archive.put(
                text, session=session_id, gist=make_gist(turn), kind=classify(text),
                turns=[self._archived_turns],
            )
            archived.append(rec.id)

        summary, usage = self._summarize(old)
        header = f"{HEADER_PREFIX}\n{summary}\n\n{self._pointer_table(session_id)}"
        return CompactionResult(
            messages=[prepend_text(kept[0], header), *kept[1:]],
            removed_messages=len(old),
            usage=usage,
            archived_ids=archived,
        )

    def _pointer_table(self, session_id: str) -> str:
        records = self._archive.records(session_id)
        shown = records[-self._max_pointers:]
        lines = ["[Archived turns: exact text is available through the recall tool]"]
        lines += [f"§{r.id} → {r.gist}" for r in shown]
        if hidden := len(records) - len(shown):
            lines.append(f"({hidden} older archived turns: find them with a recall query)")
        return "\n".join(lines)
