"""Builds sessions for each strategy, so the CLI, server and benchmark agree."""

from __future__ import annotations

import uuid

from rewind.compaction import NullCompactor, PlainCompactor, RewindCompactor
from rewind.config import Settings
from rewind.llm import LLM
from rewind.recall import RecallTool
from rewind.session import Session
from rewind.store import ArchiveStore

STRATEGIES = ("none", "plain", "rewind")


def build_session(strategy: str, settings: Settings, llm: LLM,
                  archive: ArchiveStore | None = None, session_id: str | None = None,
                  context_limit: int | None = None) -> Session:
    sid = session_id or uuid.uuid4().hex[:12]
    common = dict(session_id=sid, llm=llm, context_limit=context_limit or settings.context_limit,
                  max_output_tokens=settings.max_output_tokens)
    if strategy == "none":
        return Session(compactor=NullCompactor(), **common)
    if strategy == "plain":
        return Session(compactor=PlainCompactor(llm, settings.keep_recent_messages,
                                                settings.summary_max_tokens), **common)
    if strategy == "rewind":
        if archive is None:
            raise ValueError("the rewind strategy needs an archive")
        return Session(
            compactor=RewindCompactor(llm, archive, settings.keep_recent_messages,
                                      settings.summary_max_tokens, settings.max_pointers),
            tools=RecallTool(archive, sid, min_coverage=settings.recall_min_coverage),
            **common,
        )
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
