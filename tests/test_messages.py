from rewind.messages import find_cut, is_plain_user, prepend_text, to_text

from fakes import FakeTextBlock


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": [FakeTextBlock(text)]}


TOOL_USE = {"role": "assistant", "content": [
    {"type": "tool_use", "id": "t1", "name": "recall", "input": {"id": "abc"}}]}
TOOL_RESULT = {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t1", "content": "archived text"}]}


def test_to_text_handles_strings_sdk_blocks_and_tools():
    assert to_text(user("hi")) == "user: hi"
    assert to_text(assistant("hello")) == "assistant: hello"
    assert "[called recall({'id': 'abc'})]" in to_text(TOOL_USE)
    assert "[tool result: archived text]" in to_text(TOOL_RESULT)


def test_to_text_skips_thinking_blocks():
    msg = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "secret"}, {"type": "text", "text": "answer"}]}
    assert to_text(msg) == "assistant: answer"


def test_tool_results_are_not_plain_user_messages():
    assert is_plain_user(user("hi"))
    assert not is_plain_user(TOOL_RESULT)
    assert not is_plain_user(assistant("x"))


def test_find_cut_keeps_recent_messages():
    msgs = [user("1"), assistant("1"), user("2"), assistant("2"), user("3"), assistant("3")]
    cut = find_cut(msgs, keep_recent=2)
    assert cut == 4
    assert len(msgs[cut:]) >= 2


def test_find_cut_never_splits_tool_use_from_result():
    msgs = [user("1"), assistant("1"), user("2"), TOOL_USE, TOOL_RESULT, assistant("2"),
            user("3"), assistant("3")]
    for keep in range(1, len(msgs)):
        cut = find_cut(msgs, keep_recent=keep)
        if cut:
            assert is_plain_user(msgs[cut])
            assert msgs[cut - 1] is not TOOL_USE


def test_find_cut_returns_zero_when_nothing_to_compact():
    assert find_cut([user("1"), assistant("1")], keep_recent=4) == 0
    assert find_cut([], keep_recent=0) == 0


def test_prepend_text_keeps_original_content():
    out = prepend_text(user("question"), "SUMMARY")
    assert out["content"].startswith("SUMMARY")
    assert out["content"].endswith("question")
    blocks = prepend_text({"role": "user", "content": [{"type": "text", "text": "q"}]}, "S")
    assert blocks["content"][0] == {"type": "text", "text": "S"}
