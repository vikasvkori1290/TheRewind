"""Token accounting shared by the session and the compactors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0

    def add_response(self, response: Any) -> None:
        u = response.usage
        self.input_tokens += u.input_tokens or 0
        self.output_tokens += u.output_tokens or 0
        self.cache_read_tokens += getattr(u, "cache_read_input_tokens", None) or 0
        self.cache_write_tokens += getattr(u, "cache_creation_input_tokens", None) or 0
        self.calls += 1

    def add(self, other: Usage) -> None:
        for field, value in asdict(other).items():
            setattr(self, field, getattr(self, field) + value)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True)
class CompactionEvent:
    turn: int
    strategy: str
    removed_messages: int
    tokens_before: int
    tokens_after: int
