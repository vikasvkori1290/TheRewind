"""Opens the archive backend chosen in settings."""

from __future__ import annotations

from pathlib import Path

from rewind.config import Settings
from rewind.store import ArchiveStore, JsonlArchive


def open_archive(settings: Settings, data_dir: str | Path | None = None) -> ArchiveStore:
    path = Path(data_dir or settings.data_dir)
    if settings.store == "jsonl":
        return JsonlArchive(path)
    if settings.store == "sqlite":
        from rewind.store_sqlite import SqliteArchive

        return SqliteArchive(path)
    raise ValueError(f"unknown store {settings.store!r}; use 'jsonl' or 'sqlite'")
