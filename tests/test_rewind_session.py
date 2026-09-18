"""End-to-end Rewind behaviour with a scripted fake model."""

import re

from rewind.config import Settings
from rewind.factory import build_session
from rewind.session import REFUSAL_REPLY, TOOL_LIMIT_REPLY
from rewind.store import JsonlArchive

from fakes import FakeLLM, fake_response, fake_tool_call

CODE = "def compute_late_fee(days_overdue, daily_rate=0.75):\n    return min(days_overdue * daily_rate, 30.0)"
SETTINGS = Settings(context_limit=120, keep_recent_messages=2)


def fill(session, n):
    for i in range(n):
        session.send(f"filler message {i} " + "lorem ipsum " * 10)


def pointer_for(session, needle):
    header = session.messages[0]["content"]
    line = next(ln for ln in header.splitlines() if needle in ln)
    return re.search(r"§([0-9a-f]{12})", line).group(1)


def test_planted_code_is_recalled_verbatim_after_compaction(tmp_path):
    llm = FakeLLM()
    session = build_session("rewind", SETTINGS, llm, archive=JsonlArchive(tmp_path), session_id="s")
    session.send(f"Keep this function:\n{CODE}")
    fill(session, 6)
    assert session.compactions, "planted turn should have been compacted"
    assert CODE not in str(session.messages[1:])

    rid = pointer_for(session, "compute_late_fee")
    llm.replies = [fake_tool_call({"id": rid, "level": "full"}),
                   lambda msgs: "Here it is:\n" + msgs[-1]["content"][0]["content"]]
    answer = session.send("Show me the exact compute_late_fee function from earlier.")

    assert CODE in answer
    assert session.tools.stats.hits_by_id == 1
    # tool_use and tool_result stay paired in history
    assert session.messages[-3]["content"][1].type == "tool_use"
    assert session.messages[-2]["content"][0]["type"] == "tool_result"


def test_rewind_sends_recall_tool_and_system_prompt(tmp_path):
    llm = FakeLLM()
    session = build_session("rewind", SETTINGS, llm, archive=JsonlArchive(tmp_path))
    session.send("hi")
    call = llm.calls[-1]
    assert call["tools"][0]["name"] == "recall"
    assert "recall tool" in call["system"]


def test_plain_session_has_no_tools(tmp_path):
    llm = FakeLLM()
    session = build_session("plain", SETTINGS, llm)
    session.send("hi")
    assert llm.calls[-1]["tools"] is None


def test_tool_loop_is_bounded(tmp_path):
    llm = FakeLLM(replies=[fake_tool_call({"query": f"thing {i}"}, f"c{i}") for i in range(10)])
    session = build_session("rewind", SETTINGS, llm, archive=JsonlArchive(tmp_path))
    assert session.send("loop please") == TOOL_LIMIT_REPLY


def test_refusal_mid_tool_loop_rolls_back_the_turn(tmp_path):
    llm = FakeLLM(replies=[fake_tool_call({"query": "x"}), fake_response("", stop_reason="refusal")])
    session = build_session("rewind", SETTINGS, llm, archive=JsonlArchive(tmp_path))
    session.send("first")
    before = list(session.messages)
    llm.replies = [fake_tool_call({"query": "x"}), fake_response("", stop_reason="refusal")]
    assert session.send("second") == REFUSAL_REPLY
    assert session.messages == before


def test_rewind_requires_archive():
    import pytest
    with pytest.raises(ValueError):
        build_session("rewind", SETTINGS, FakeLLM())
