"""Helpers for working with Messages API conversation history."""

from __future__ import annotations

import re
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


HEADER_PREFIX = "[Earlier conversation, compacted]"
_HEADER_SEPARATOR = "\n\n---\n\n"


def prepend_text(message: dict, header: str) -> dict:
    """Return a copy of a plain user message with `header` placed before its content."""
    content = message["content"]
    if isinstance(content, str):
        return {"role": "user", "content": f"{header}{_HEADER_SEPARATOR}{content}"}
    return {"role": "user", "content": [{"type": "text", "text": header}, *content]}


def strip_header(message: dict) -> dict:
    """Undo `prepend_text` for a compaction header, so archives hold only real turns."""
    content = message["content"]
    if isinstance(content, str):
        if content.startswith(HEADER_PREFIX) and _HEADER_SEPARATOR in content:
            return {**message, "content": content.split(_HEADER_SEPARATOR, 1)[1]}
        return message
    blocks = [_as_dict(b) for b in content]
    if blocks and blocks[0].get("type") == "text" and blocks[0]["text"].startswith(HEADER_PREFIX):
        return {**message, "content": blocks[1:]}
    return message


def split_turns(messages: list[dict]) -> list[list[dict]]:
    """Group history into turns: each starts at a plain user message and runs to the next."""
    turns: list[list[dict]] = []
    for m in messages:
        if is_plain_user(m) or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    return turns


_DEFINITION = re.compile(r"\b(?:def|class|function|fn|func|interface|struct)\s+([A-Za-z_]\w*)")


def classify(text: str) -> str:
    if "```" in text or _DEFINITION.search(text):
        return "code"
    if "[called " in text:
        return "tool"
    return "text"


def make_gist(turn: list[dict], max_len: int = 90) -> str:
    """One line naming what a turn was about: the user's first line plus defined names."""
    user_text = to_text(strip_header(turn[0])).removeprefix("user: ")
    first_line = next((ln.strip() for ln in user_text.splitlines() if ln.strip()), "")
    names = list(dict.fromkeys(_DEFINITION.findall("\n".join(to_text(m) for m in turn))))
    gist = first_line if len(first_line) <= max_len else first_line[: max_len - 1] + "…"
    if names:
        gist += f" [{', '.join(names[:3])}]"
    return gist or "(empty turn)"
