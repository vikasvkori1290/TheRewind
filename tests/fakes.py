"""Offline stand-ins for the LLM gateway."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

from rewind.messages import to_text


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def model_dump(self) -> dict:
        return asdict(self)


def _usage(input_tokens=10, output_tokens=5):
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                           cache_read_input_tokens=0, cache_creation_input_tokens=None)


def fake_response(text: str, input_tokens: int = 10, output_tokens: int = 5,
                  stop_reason: str = "end_turn"):
    return SimpleNamespace(content=[FakeTextBlock(text)] if text else [],
                           stop_reason=stop_reason, usage=_usage(input_tokens, output_tokens))


def fake_tool_call(tool_input: dict, call_id: str = "call_1", name: str = "recall"):
    return SimpleNamespace(
        content=[FakeTextBlock("Let me check."), FakeToolUseBlock(call_id, name, tool_input)],
        stop_reason="tool_use", usage=_usage())


@dataclass
class FakeLLM:
    """Counts roughly 1 token per 4 characters.

    Replies come from `replies` in order: a string, a response object, or a
    callable taking the request messages. With no replies left it echoes.
    """

    model: str = "fake-model"
    replies: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def create(self, *, system, messages, max_tokens, tools=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        is_summary = "Summarize the conversation" in str(messages[0]["content"])[:200]
        if self.replies and not is_summary:
            reply = self.replies.pop(0)
            if callable(reply):
                reply = reply(messages)
            return fake_response(reply) if isinstance(reply, str) else reply
        return fake_response("summary of earlier turns" if is_summary else f"reply {len(self.calls)}")

    def count_tokens(self, *, system, messages, tools=None):
        chars = len(system) + sum(len(to_text(m)) for m in messages)
        return chars // 4
