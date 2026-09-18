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


def active_session(tmp_path, llm, sid="s"):
    """A Rewind session whose archive already has a turn, so its tools are active."""
    archive = JsonlArchive(tmp_path)
    archive.put("user: earlier turn", session=sid, gist="earlier")
    return build_session("rewind", SETTINGS, llm, archive=archive, session_id=sid)


def test_recall_tool_is_only_sent_after_the_first_compaction(tmp_path):
    llm = FakeLLM()
    session = build_session("rewind", SETTINGS, llm, archive=JsonlArchive(tmp_path), session_id="s")
    session.send("hi")
    assert llm.calls[-1]["tools"] is None
    assert "recall tool" not in llm.calls[-1]["system"]
    fill(session, 6)
    assert session.compactions
    call = llm.calls[-1]
    assert call["tools"][0]["name"] == "recall"
    assert "recall tool" in call["system"]


def test_named_identifier_is_attached_without_a_tool_call(tmp_path):
    llm = FakeLLM()
    session = build_session("rewind", SETTINGS, llm, archive=JsonlArchive(tmp_path), session_id="s")
    session.send(f"Keep this function:\n{CODE}")
    fill(session, 6)
    def chat_calls():  # model calls excluding compaction summaries
        return [c for c in llm.calls if "Summarize the conversation" not in str(c["messages"][0])]

    before = len(chat_calls())
    session.send("Show me compute_late_fee again.")
    sent = llm.calls[-1]["messages"][-1]["content"]
    assert CODE in sent and "attached automatically" in sent
    assert len(chat_calls()) == before + 1  # one model call, no recall round trip
    assert session.tools.stats.prefetches == 1
    session.send("And compute_late_fee once more?")  # already attached: not repeated
    assert session.tools.stats.prefetches == 1


def test_plain_session_has_no_tools(tmp_path):
    llm = FakeLLM()
    session = build_session("plain", SETTINGS, llm)
    session.send("hi")
    assert llm.calls[-1]["tools"] is None


def test_tool_loop_is_bounded(tmp_path):
    llm = FakeLLM(replies=[fake_tool_call({"query": f"thing {i}"}, f"c{i}") for i in range(10)])
    session = active_session(tmp_path, llm)
    assert session.send("loop please") == TOOL_LIMIT_REPLY


def test_refusal_mid_tool_loop_rolls_back_the_turn(tmp_path):
    llm = FakeLLM(replies=[fake_tool_call({"query": "x"}), fake_response("", stop_reason="refusal")])
    session = active_session(tmp_path, llm)
    session.send("first")
    before = list(session.messages)
    llm.replies = [fake_tool_call({"query": "x"}), fake_response("", stop_reason="refusal")]
    assert session.send("second") == REFUSAL_REPLY
    assert session.messages == before


def test_rewind_requires_archive():
    import pytest
    with pytest.raises(ValueError):
        build_session("rewind", SETTINGS, FakeLLM())
