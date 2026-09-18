"""The `recall` tool: lets the model fetch archived turns by ID or by search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from rewind.search import query_terms
from rewind.store import ArchiveStore, SearchHit, is_valid_id

RECALL_SYSTEM = (
    "Parts of this conversation may have been compacted. The start of the history "
    "then holds a summary and a list of archived turns. When the user asks about "
    "earlier details you cannot see verbatim (code, numbers, names, decisions), "
    "call the recall tool instead of guessing or rewriting them from memory."
)

RECALL_TOOL = {
    "name": "recall",
    "description": (
        "Fetch conversation turns that were compacted out of context. Pass `id` "
        "(from the [Archived turns] list, without the § sign) when one matches, "
        "otherwise pass `query` with distinctive keywords such as names, numbers "
        "or identifiers. Use level 'gist' to check what an item is and 'full' when "
        "you need the exact original text. NOT_FOUND means the content was never "
        "archived: answer from what you know or produce it fresh, and do not "
        "repeat the same search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Archive id, 12 hex characters"},
            "query": {"type": "string", "description": "Keywords to search the archive for"},
            "level": {"type": "string", "enum": ["gist", "full"],
                      "description": "gist (default) or full original text"},
        },
    },
}

NOT_FOUND = "NOT_FOUND: nothing in the archive matches. Answer without it or generate it fresh."


@dataclass(frozen=True)
class ToolOutcome:
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class RecallEvent:
    turn: int
    outcome: str  # hit_id | hit_search | miss | repeat_miss | invalid
    level: str
    id: str | None = None
    query: str | None = None
    record_ids: tuple[str, ...] = ()
    chars_returned: int = 0


@dataclass
class RecallStats:
    calls: int = 0
    hits_by_id: int = 0
    hits_by_search: int = 0
    misses: int = 0
    repeated_misses: int = 0
    invalid: int = 0
    chars_returned: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _quote(record_id: str, kind: str, text: str) -> str:
    # Archived text is data from earlier in the conversation, never instructions;
    # neutralize anything that would close the wrapper early.
    body = text.replace("</archived>", "<\\/archived>")
    return f'<archived id="{record_id}" kind="{kind}">\n{body}\n</archived>'


@dataclass
class RecallTool:
    archive: ArchiveStore
    session_id: str
    min_coverage: float = 0.5
    max_results: int = 3
    stats: RecallStats = field(default_factory=RecallStats)
    events: list[RecallEvent] = field(default_factory=list)
    _failed: set[tuple[str, ...]] = field(default_factory=set, init=False, repr=False)

    @property
    def definitions(self) -> list[dict]:
        return [RECALL_TOOL]

    def handle(self, name: str, tool_input: dict, turn: int) -> ToolOutcome:
        if name != RECALL_TOOL["name"]:
            return ToolOutcome(f"Unknown tool: {name}", is_error=True)
        self.stats.calls += 1
        level = "full" if tool_input.get("level") == "full" else "gist"
        cid = str(tool_input.get("id") or "").strip().lstrip("§").lower()
        query = str(tool_input.get("query") or "").strip()

        if cid:
            record = self.archive.get(cid) if is_valid_id(cid) else None
            if record and record.session == self.session_id:
                text = self._render_record(record, level)
                return self._record(turn, "hit_id", level, text, id=cid, record_ids=(cid,))
            if not query:
                self.stats.invalid += 1
                return self._record(turn, "invalid", level,
                                    f"NOT_FOUND: no archived turn has id {cid!r}.", id=cid)

        if not query:
            self.stats.invalid += 1
            return self._record(turn, "invalid", level, "Pass either an id or a query.")
        return self._search(query, level, turn)

    def _search(self, query: str, level: str, turn: int) -> ToolOutcome:
        key = tuple(sorted(query_terms(query)))
        if key in self._failed:
            self.stats.repeated_misses += 1
            return self._record(turn, "repeat_miss", level, NOT_FOUND, query=query)

        hits = [h for h in self.archive.search(query, session=self.session_id,
                                               k=self.max_results)
                if h.coverage >= self.min_coverage]
        if not hits:
            self._failed.add(key)
            self.stats.misses += 1
            return self._record(turn, "miss", level, NOT_FOUND, query=query)

        self.stats.hits_by_search += 1
        text = self._render_hits(hits, level)
        return self._record(turn, "hit_search", level, text, query=query,
                            record_ids=tuple(h.record.id for h in hits))

    def _render_record(self, record, level: str) -> str:
        if level == "full":
            return _quote(record.id, record.kind, self.archive.text(record.id) or "")
        return (f"§{record.id} ({record.kind}, {record.chars} chars): {record.gist}\n"
                f"Call recall with this id and level 'full' for the exact text.")

    def _render_hits(self, hits: list[SearchHit], level: str) -> str:
        lines = [f"§{h.record.id} ({h.record.kind}): {h.record.gist}" for h in hits]
        if level == "full":
            best = hits[0].record
            out = _quote(best.id, best.kind, self.archive.text(best.id) or "")
            if len(lines) > 1:
                out += "\nOther matches:\n" + "\n".join(lines[1:])
            return out
        return "Matches (use an id with level 'full' for exact text):\n" + "\n".join(lines)

    def _record(self, turn: int, outcome: str, level: str, text: str, **kw) -> ToolOutcome:
        self.stats.chars_returned += len(text)
        self.events.append(RecallEvent(turn=turn, outcome=outcome, level=level,
                                       chars_returned=len(text), **kw))
        if outcome == "hit_id":
            self.stats.hits_by_id += 1
        return ToolOutcome(text)
