"""SQLite archive: one file, transactional writes, FTS5 keyword search.

Same interface and behaviour as `JsonlArchive`; choose it with REWIND_STORE=sqlite.
FTS5 ranks candidates by BM25; coverage is computed the same way as for JSONL.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from rewind.search import coverage, query_terms, tokenize
from rewind.store import Record, SearchHit, _now, content_id, is_valid_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    session TEXT NOT NULL,
    gist TEXT NOT NULL,
    kind TEXT NOT NULL,
    turns TEXT NOT NULL,
    chars INTEGER NOT NULL,
    created TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_session ON records(session);
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    id UNINDEXED, session UNINDEXED, terms
);
"""


def _row_to_record(row) -> Record:
    rid, session, gist, kind, turns, chars, created = row
    return Record(id=rid, session=session, gist=gist, kind=kind, turns=json.loads(turns),
                  chars=chars, created=created)


class SqliteArchive:
    def __init__(self, data_dir: str | Path):
        path = Path(data_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path / "archive.sqlite3", check_same_thread=False)
        self._lock = threading.Lock()
        try:
            self._db.executescript(_SCHEMA)
        except sqlite3.OperationalError as e:
            raise RuntimeError("this SQLite build lacks FTS5; use REWIND_STORE=jsonl") from e

    def put(self, text, *, session, gist, kind="text", turns=None):
        cid = content_id(session, text)
        with self._lock, self._db:
            if existing := self._get(cid):
                return existing
            rec = Record(id=cid, session=session, gist=gist, kind=kind, turns=list(turns or []),
                         chars=len(text), created=_now())
            self._db.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rec.id, rec.session, rec.gist, rec.kind, json.dumps(rec.turns), rec.chars,
                 rec.created, text))
            # Index our own normalized tokens so search matches JsonlArchive exactly.
            self._db.execute("INSERT INTO records_fts VALUES (?, ?, ?)",
                             (rec.id, rec.session, " ".join(tokenize(f"{gist}\n{text}"))))
            return rec

    def get(self, cid):
        if not is_valid_id(cid):
            return None
        with self._lock:
            return self._get(cid)

    def _get(self, cid):
        row = self._db.execute(
            "SELECT id, session, gist, kind, turns, chars, created FROM records WHERE id = ?",
            (cid,)).fetchone()
        return _row_to_record(row) if row else None

    def text(self, cid):
        if not is_valid_id(cid):
            return None
        with self._lock:
            row = self._db.execute("SELECT body FROM records WHERE id = ?", (cid,)).fetchone()
        return row[0] if row else None

    def records(self, session):
        with self._lock:
            rows = self._db.execute(
                "SELECT id, session, gist, kind, turns, chars, created FROM records "
                "WHERE session = ? ORDER BY rowid", (session,)).fetchall()
        return [_row_to_record(r) for r in rows]

    def search(self, query, *, session, k=3):
        terms = query_terms(query)
        if not terms:
            return []
        # Terms are [a-z0-9_.] only, so quoting each one is a safe FTS5 query.
        match = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            rows = self._db.execute(
                "SELECT r.id, r.session, r.gist, r.kind, r.turns, r.chars, r.created, "
                "f.terms, -bm25(records_fts) FROM records_fts f JOIN records r ON r.id = f.id "
                "WHERE records_fts MATCH ? AND f.session = ? ORDER BY bm25(records_fts) LIMIT ?",
                (match, session, max(k * 5, 20))).fetchall()
        hits = [SearchHit(_row_to_record(row[:7]), row[8], coverage(terms, set(row[7].split())))
                for row in rows]
        hits.sort(key=lambda h: (h.coverage, h.score), reverse=True)
        return hits[:k]
