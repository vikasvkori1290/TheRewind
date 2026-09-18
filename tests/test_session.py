from rewind.compaction import PlainCompactor
from rewind.session import REFUSAL_REPLY, Session

from fakes import FakeLLM, fake_response


def make_session(llm, context_limit=10_000, keep_recent=2):
    compactor = PlainCompactor(llm, keep_recent=keep_recent, summary_max_tokens=500)
    return Session("s1", llm, compactor, context_limit=context_limit, max_output_tokens=1000)


def test_send_returns_reply_and_records_usage():
    llm = FakeLLM(replies=["hello there"])
    session = make_session(llm)
    assert session.send("hi") == "hello there"
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    assert session.usage.calls == 1
    assert session.usage.input_tokens == 10 and session.usage.output_tokens == 5


def test_no_compaction_under_limit():
    session = make_session(FakeLLM())
    for i in range(5):
        session.send(f"message {i}")
    assert session.compactions == []
    assert len(session.messages) == 10


def test_compacts_when_over_limit_and_keeps_recent_turns():
    llm = FakeLLM()
    session = make_session(llm, context_limit=60, keep_recent=2)
    for i in range(6):
        session.send(f"message number {i} " + "x" * 40)

    assert session.compactions, "expected at least one compaction"
    event = session.compactions[0]
    assert event.strategy == "plain"
    assert event.tokens_after < event.tokens_before
    # history still starts with a user turn that carries the summary
    first = session.messages[0]
    assert first["role"] == "user"
    assert first["content"].startswith("[Earlier conversation, compacted]")
    # the latest user message is never compacted away
    assert "message number 5" in session.messages[-2]["content"]


def test_compaction_cost_is_counted_separately():
    llm = FakeLLM()
    session = make_session(llm, context_limit=60, keep_recent=2)
    for i in range(6):
        session.send(f"message number {i} " + "x" * 40)
    assert session.compaction_usage.calls == len(session.compactions)
    total = session.total_usage
    assert total.calls == session.usage.calls + session.compaction_usage.calls


def test_summary_request_contains_old_turns_only():
    llm = FakeLLM()
    session = make_session(llm, context_limit=60, keep_recent=2)
    for i in range(6):
        session.send(f"message number {i} " + "x" * 40)
    summary_calls = [c for c in llm.calls if "Summarize" in str(c["messages"][0]["content"])]
    assert summary_calls
    prompt = summary_calls[0]["messages"][0]["content"]
    assert "message number 0" in prompt


def test_refusal_keeps_history_valid():
    llm = FakeLLM(replies=[fake_response("", stop_reason="refusal")])
    session = make_session(llm)
    assert session.send("something") == REFUSAL_REPLY
    assert session.messages == []
