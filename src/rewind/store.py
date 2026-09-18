"""Archive of compacted conversation text.

Text is stored verbatim under a content ID and never modified. Everything that
touches the archive depends on the `ArchiveStore` protocol, so the JSONL
implementation here and the SQLite one in `store_sqlite.py` are interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from rewind.search import BM25, coverage, query_terms, tokenize

# IDs come back from the model inside tool calls, so they are validated before
# they are ever used to build a file path.
ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")


def content_id(session: str, text: str) -> str:
    # The session is part of the hash so identical text in two sessions never
    # shares a record, which keeps sessions isolated.
    return hashlib.sha256(f"{session}\n{text}".encode()).hexdigest()[:12]


def is_valid_id(cid: str) -> bool:
    return bool(ID_PATTERN.match(cid))


@dataclass(frozen=True)
class Record:
    id: str
    session: str
    gist: str
    kind: str = "text"
    turns: list[int] = field(default_factory=list)
    chars: int = 0
    created: str = ""


@dataclass(frozen=True)
class SearchHit:
    record: Record
    score: float
    coverage: float


class ArchiveStore(Protocol):
    def put(self, text: str, *, session: str, gist: str, kind: str = "text",
            turns: list[int] | None = None) -> Record: ...

    def get(self, cid: str) -> Record | None: ...

    def text(self, cid: str) -> str | None: ...

    def records(self, session: str) -> list[Record]: ...

    def search(self, query: str, *, session: str, k: int = 3) -> list[SearchHit]: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JsonlArchive:
    """Append-only `archive.jsonl` for metadata plus one file per text in `objects/`."""

    def __init__(self, data_dir: str | Path):
        self._dir = Path(data_dir)
        self._objects = self._dir / "objects"
        self._objects.mkdir(parents=True, exist_ok=True)
        self._log = self._dir / "archive.jsonl"
        self._lock = threading.Lock()
        self._records: dict[str, Record] = {}
        self._index: dict[str, tuple[list[Record], BM25, list[set[str]]]] = {}
        if self._log.exists():
            for line in self._log.read_text().splitlines():
                if line.strip():
                    rec = Record(**json.loads(line))
                    self._records[rec.id] = rec

    def put(self, text, *, session, gist, kind="text", turns=None):
        cid = content_id(session, text)
        with self._lock:
            if cid in self._records:
                return self._records[cid]
            # Object first, then the log line: a crash can leave an orphan
            # object, never a log entry that points at missing text.
            (self._objects / f"{cid}.txt").write_text(text)
            rec = Record(id=cid, session=session, gist=gist, kind=kind,
                         turns=list(turns or []), chars=len(text), created=_now())
            with self._log.open("a") as f:
                f.write(json.dumps(asdict(rec)) + "\n")
            self._records[cid] = rec
            self._index.pop(session, None)
            return rec

    def get(self, cid):
        return self._records.get(cid) if is_valid_id(cid) else None

    def text(self, cid):
        if not self.get(cid):
            return None
        path = self._objects / f"{cid}.txt"
        return path.read_text() if path.exists() else None

    def records(self, session):
        return [r for r in self._records.values() if r.session == session]

    def search(self, query, *, session, k=3):
        terms = query_terms(query)
        if not terms:
            return []
        with self._lock:
            index = self._index.get(session) or self._build_index(session)
        recs, bm25, token_sets = index
        scored = [
            SearchHit(rec, score, coverage(terms, toks))
            for rec, score, toks in zip(recs, bm25.scores(terms), token_sets)
            if score > 0
        ]
        scored.sort(key=lambda h: (h.coverage, h.score), reverse=True)
        return scored[:k]

    def _build_index(self, session):
        recs = self.records(session)
        docs = [tokenize(f"{r.gist}\n{self.text(r.id) or ''}") for r in recs]
        index = (recs, BM25(docs), [set(d) for d in docs])
        self._index[session] = index
        return index
