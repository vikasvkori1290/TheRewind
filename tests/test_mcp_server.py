import asyncio
import re

import pytest

from rewind.mcp_server import MAX_REMEMBER_CHARS, MemoryTools, build_server
from rewind.store import JsonlArchive


@pytest.fixture
def tools(tmp_path):
    return MemoryTools(JsonlArchive(tmp_path), "proj")


def test_remember_then_recall_by_id_and_query(tools):
    saved = tools.remember("def add(a, b):\n    return a + b", "add helper")
    rid = re.search(r"§([0-9a-f]{12})", saved).group(1)
    assert "(code," in saved
    assert "return a + b" in tools.recall(id=rid, level="full")
    assert "add helper" in tools.recall(query="add helper")
    assert rid in tools.list_items()


def test_remember_validation(tools):
    assert "empty" in tools.remember("   ", "x")
    assert "too long" in tools.remember("x" * (MAX_REMEMBER_CHARS + 1), "x")
    assert "first line" in tools.remember("first line\nsecond", "")


def test_projects_are_isolated(tmp_path):
    archive = JsonlArchive(tmp_path)
    MemoryTools(archive, "a").remember("secret alpha plan", "alpha")
    assert "NOT_FOUND" in MemoryTools(archive, "b").recall(query="alpha plan")


def test_server_registers_tools(tools):
    pytest.importorskip("mcp")
    server = build_server(tools)
    listed = asyncio.run(server.list_tools())
    assert {t.name for t in listed} == {"remember", "recall", "list_memories"}
