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


def fake_response(text: str, input_tokens: int = 10, output_tokens: int = 5,
                  stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[FakeTextBlock(text)] if text else [],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=None,
        ),
    )


@dataclass
class FakeLLM:
    """Counts roughly 1 token per 4 characters; replies from a script or echoes."""

    model: str = "fake-model"
    replies: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def create(self, *, system, messages, max_tokens, tools=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if self.replies:
            reply = self.replies.pop(0)
            return reply if not isinstance(reply, str) else fake_response(reply)
        return fake_response(f"reply {len(self.calls)}")

    def count_tokens(self, *, system, messages, tools=None):
        chars = len(system) + sum(len(to_text(m)) for m in messages)
        return chars // 4
