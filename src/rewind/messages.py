"""Helpers for working with Messages API conversation history."""

from __future__ import annotations

from typing import Any


def _as_dict(block: Any) -> dict:
    return block if isinstance(block, dict) else block.model_dump()


def to_text(message: dict) -> str:
    """Render one message as plain text, for summaries and the archive."""
    content = message["content"]
    if isinstance(content, str):
        return f"{message['role']}: {content}"
    parts = []
    for block in map(_as_dict, content):
        kind = block.get("type")
        if kind == "text":
            parts.append(block["text"])
        elif kind == "tool_use":
            parts.append(f"[called {block['name']}({block['input']})]")
        elif kind == "tool_result":
            parts.append(f"[tool result: {block.get('content')}]")
        # thinking, fallback and other blocks carry no conversation content
    return f"{message['role']}: " + "\n".join(parts)


def response_text(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def is_plain_user(message: dict) -> bool:
    """A user message that is not a tool result, so history may start there."""
    if message["role"] != "user":
        return False
    content = message["content"]
    if isinstance(content, str):
        return True
    return all(_as_dict(b).get("type") == "text" for b in content)


def find_cut(messages: list[dict], keep_recent: int) -> int:
    """Index where compaction may split history into [old | kept].

    Returns the latest plain user message that still leaves at least
    `keep_recent` messages kept, or 0 when no safe cut exists. Cutting only at
    plain user messages guarantees a tool_use is never separated from its
    tool_result.
    """
    for i in range(len(messages) - keep_recent, 0, -1):
        if is_plain_user(messages[i]):
            return i
    return 0


def prepend_text(message: dict, header: str) -> dict:
    """Return a copy of a plain user message with `header` placed before its content."""
    content = message["content"]
    if isinstance(content, str):
        return {"role": "user", "content": f"{header}\n\n---\n\n{content}"}
    return {"role": "user", "content": [{"type": "text", "text": header}, *content]}
