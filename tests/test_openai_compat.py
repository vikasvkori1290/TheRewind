import json
from types import SimpleNamespace

from rewind.config import Settings
from rewind.openai_compat import (OpenAICompatLLM, TextBlock, ToolUseBlock, from_openai_response,
                                  to_openai_messages, to_openai_tools)
from rewind.recall import RECALL_TOOL


def oa_response(content=None, tool_calls=None, finish="stop", prompt=100, completion=20, cached=0):
    calls = [SimpleNamespace(id=i, function=SimpleNamespace(name=n, arguments=a))
             for i, n, a in (tool_calls or [])]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls or None),
                                 finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached)))


def test_tools_are_converted_to_functions():
    [tool] = to_openai_tools([RECALL_TOOL])
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "recall"
    assert tool["function"]["parameters"] == RECALL_TOOL["input_schema"]
    assert to_openai_tools(None) is None


def test_messages_round_trip_tool_calls_and_results():
    history = [
        {"role": "user", "content": "show the function"},
        {"role": "assistant", "content": [TextBlock("Checking."),
                                          ToolUseBlock("call_1", "recall", {"id": "abc"})]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1",
                                      "content": "def f(): ...", "is_error": False}]},
        {"role": "assistant", "content": [TextBlock("Here it is.")]},
    ]
    out = to_openai_messages("sys", history)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool", "assistant"]
    call = out[2]["tool_calls"][0]
    assert call["id"] == "call_1" and json.loads(call["function"]["arguments"]) == {"id": "abc"}
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": "def f(): ..."}


def test_response_with_tool_call_maps_to_tool_use():
    r = from_openai_response(oa_response(None, [("c1", "recall", '{"id": "\\u00a7abc"}')],
                                         finish="tool_calls"))
    assert r.stop_reason == "tool_use"
    block = r.content[0]
    assert block.type == "tool_use" and block.input == {"id": "§abc"}


def test_response_text_usage_and_stop_reasons():
    r = from_openai_response(oa_response("hi", prompt=100, completion=7, cached=40))
    assert r.content[0].text == "hi" and r.stop_reason == "end_turn"
    assert (r.usage.input_tokens, r.usage.cache_read_input_tokens, r.usage.output_tokens) == (60, 40, 7)
    assert from_openai_response(oa_response("x", finish="length")).stop_reason == "max_tokens"
    assert from_openai_response(oa_response("x", finish="content_filter")).stop_reason == "refusal"


def test_bad_tool_arguments_become_empty_input():
    r = from_openai_response(oa_response(None, [("c1", "recall", "{not json")]))
    assert r.content[0].input == {}


def test_llm_sends_converted_request_and_estimates_tokens():
    sent = {}

    class Completions:
        def create(self, **kw):
            sent.update(kw)
            return oa_response("ok")

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    llm = OpenAICompatLLM(Settings(provider="nim", model="openai/gpt-oss-20b"), client)
    r = llm.create(system="s", messages=[{"role": "user", "content": "hi"}], max_tokens=50,
                   tools=[RECALL_TOOL])
    assert r.content[0].text == "ok"
    assert sent["model"] == "openai/gpt-oss-20b" and sent["tools"][0]["type"] == "function"
    assert llm.count_tokens(system="", messages=[{"role": "user", "content": "x" * 400}]) >= 100


def test_nim_reads_nvidia_api_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REWIND_API_KEY", raising=False)
    monkeypatch.setenv("REWIND_PROVIDER", "nim")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    assert Settings.from_env().api_key == "nvapi-test"
