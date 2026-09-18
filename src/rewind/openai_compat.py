"""Adapter for OpenAI-compatible chat APIs such as NVIDIA NIM.

The rest of Rewind speaks the Anthropic Messages format (content blocks,
tool_use / tool_result, stop_reason). This module translates requests to Chat
Completions and responses back, so sessions, compaction and recall work
unchanged on any provider that implements the OpenAI protocol.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from rewind.config import Settings
from rewind.messages import _as_dict


@dataclass
class TextBlock:
    text: str
    type: str = "text"

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

    def model_dump(self) -> dict:
        return asdict(self)


def to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("input_schema", {"type": "object"})}}
            for t in tools]


def _text_of(blocks: list[dict]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(_as_dict(b).get("text", "") for b in content or [])


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        blocks = [_as_dict(b) for b in content]
        if m["role"] == "assistant":
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in blocks if b.get("type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": _text_of(blocks) or None}
            if calls:
                msg["tool_calls"] = calls
            elif msg["content"] is None:
                msg["content"] = ""
            out.append(msg)
        else:
            # Tool results must directly follow the assistant message that called them.
            for b in blocks:
                if b.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": _result_text(b.get("content"))})
            if text := _text_of(blocks):
                out.append({"role": "user", "content": text})
    return out


def _parse_arguments(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


_STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens",
                 "content_filter": "refusal", "stop": "end_turn"}


def from_openai_response(response: Any) -> SimpleNamespace:
    choice = response.choices[0]
    message = choice.message
    content: list[Any] = []
    if message.content:
        content.append(TextBlock(message.content))
    calls = message.tool_calls or []
    for call in calls:
        content.append(ToolUseBlock(call.id, call.function.name,
                                    _parse_arguments(call.function.arguments)))
    stop = "tool_use" if calls else _STOP_REASONS.get(choice.finish_reason or "stop", "end_turn")

    usage = response.usage
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    return SimpleNamespace(
        content=content,
        stop_reason=stop,
        # OpenAI counts cached tokens inside prompt_tokens; Anthropic reports them apart.
        usage=SimpleNamespace(input_tokens=prompt - cached,
                              output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                              cache_read_input_tokens=cached, cache_creation_input_tokens=0),
    )


def make_openai_client(settings: Settings, timeout: float | None = None):
    from openai import OpenAI  # optional dependency: uv sync --extra nim

    from rewind.providers import get_provider

    base_url = settings.base_url or get_provider(settings.provider).base_url
    if not base_url:
        raise ValueError(f"provider {settings.provider!r} needs a base URL")
    return OpenAI(base_url=base_url, api_key=settings.api_key or "missing",
                  timeout=timeout or settings.request_timeout)


class OpenAICompatLLM:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.model = settings.model
        # OpenAI's current models take max_completion_tokens; most other
        # compatible APIs only understand max_tokens.
        self._limit_param = "max_completion_tokens" if settings.provider == "openai" else "max_tokens"
        self._client = client or make_openai_client(settings)

    def create(self, *, system, messages, max_tokens, tools=None):
        kwargs: dict[str, Any] = dict(model=self.model, messages=to_openai_messages(system, messages))
        kwargs[self._limit_param] = max_tokens
        if oa_tools := to_openai_tools(tools):
            kwargs["tools"] = oa_tools
        return from_openai_response(self._client.chat.completions.create(**kwargs))

    def count_tokens(self, *, system, messages, tools=None):
        from rewind.llm import estimate_tokens

        # No token-counting endpoint in this protocol: estimate (~4 chars/token).
        return estimate_tokens(system, messages, tools)
