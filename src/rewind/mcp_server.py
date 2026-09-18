"""MCP server: gives any MCP client (Claude Code, desktop apps) `remember` and `recall`.

An agent saves exact text it may need after its context is compacted, then
fetches it back by ID or keyword search. Uses the same archive as the proxy.

    uv run --extra mcp rewind-mcp

Settings: REWIND_DATA_DIR (use an absolute path), REWIND_STORE, and
REWIND_MCP_SESSION to keep separate projects apart (default "mcp").
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from rewind.config import Settings
from rewind.messages import classify
from rewind.recall import RecallTool
from rewind.store import ArchiveStore
from rewind.store_factory import open_archive

MAX_REMEMBER_CHARS = 200_000
MAX_GIST_CHARS = 200


@dataclass
class MemoryTools:
    """Transport-independent tool logic, so it can be tested without MCP."""

    archive: ArchiveStore
    session: str

    def __post_init__(self):
        self._recall = RecallTool(self.archive, self.session)

    def remember(self, text: str, gist: str) -> str:
        if not text.strip():
            return "Nothing to remember: text is empty."
        if len(text) > MAX_REMEMBER_CHARS:
            return f"Text is too long ({len(text)} chars; limit {MAX_REMEMBER_CHARS})."
        gist = " ".join(gist.split())[:MAX_GIST_CHARS] or text.strip().splitlines()[0][:80]
        rec = self.archive.put(text, session=self.session, gist=gist, kind=classify(text))
        return f"Saved as §{rec.id} ({rec.kind}, {rec.chars} chars): {rec.gist}"

    def recall(self, id: str = "", query: str = "", level: str = "gist") -> str:
        return self._recall.handle("recall", {"id": id, "query": query, "level": level},
                                   turn=0).content

    def list_items(self, limit: int = 50) -> str:
        records = self.archive.records(self.session)[-limit:]
        if not records:
            return "The archive is empty."
        return "\n".join(f"§{r.id} [{r.kind}] {r.gist}" for r in records)


def build_server(tools: MemoryTools):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "rewind",
        instructions=(
            "Long-term exact memory. Use `remember` for text you may need verbatim after "
            "your context is compacted (code, numbers, decisions). Use `recall` with an id "
            "or keywords to get it back instead of regenerating it."
        ),
    )

    @server.tool(description="Save exact text under a short gist; returns its §id.")
    def remember(text: str, gist: str) -> str:
        return tools.remember(text, gist)

    @server.tool(description=(
        "Fetch saved text by id (12 hex characters, without §) or by keyword query. "
        "level 'gist' returns a one-line description; 'full' returns the exact text."))
    def recall(id: str = "", query: str = "", level: str = "gist") -> str:
        return tools.recall(id=id, query=query, level=level)

    @server.tool(description="List saved items (newest last) with their ids and gists.")
    def list_memories(limit: int = 50) -> str:
        return tools.list_items(limit)

    return server


def main() -> None:
    settings = Settings.from_env()
    tools = MemoryTools(open_archive(settings), os.getenv("REWIND_MCP_SESSION", "mcp"))
    build_server(tools).run("stdio")


if __name__ == "__main__":
    main()
