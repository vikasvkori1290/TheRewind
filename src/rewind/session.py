"""A chat session that counts tokens, compacts when over the limit, and runs tools."""

from __future__ import annotations

from typing import Protocol

from rewind.compaction import Compactor
from rewind.llm import LLM
from rewind.messages import response_text
from rewind.metrics import CompactionEvent, Usage
from rewind.recall import ToolOutcome

DEFAULT_SYSTEM = "You are a helpful assistant."
REFUSAL_REPLY = "[The model declined this request.]"
TOOL_LIMIT_REPLY = "[Stopped after too many tool calls in one turn.]"


class ToolBox(Protocol):
    @property
    def definitions(self) -> list[dict]: ...

    def handle(self, name: str, tool_input: dict, turn: int) -> ToolOutcome: ...


class Session:
    def __init__(
        self,
        session_id: str,
        llm: LLM,
        compactor: Compactor,
        context_limit: int,
        max_output_tokens: int,
        system: str = DEFAULT_SYSTEM,
        tools: ToolBox | None = None,
        max_tool_rounds: int = 5,
    ):
        self.id = session_id
        self.system = system
        self.messages: list[dict] = []
        self.usage = Usage()
        self.compaction_usage = Usage()
        self.compactions: list[CompactionEvent] = []
        self.turn = 0
        self.context_tokens = 0
        self.tools = tools
        self._llm = llm
        self._compactor = compactor
        self._context_limit = context_limit
        self._max_output_tokens = max_output_tokens
        self._max_tool_rounds = max_tool_rounds

    @property
    def strategy(self) -> str:
        return self._compactor.name

    @property
    def _tool_defs(self) -> list[dict] | None:
        return self.tools.definitions if self.tools else None

    def count_tokens(self) -> int:
        return self._llm.count_tokens(system=self.system, messages=self.messages,
                                      tools=self._tool_defs)

    def send(self, text: str) -> str:
        self.turn += 1
        self.messages.append({"role": "user", "content": text})
        self._compact_if_needed()
        turn_start = len(self.messages) - 1

        for _ in range(self._max_tool_rounds + 1):
            response = self._llm.create(system=self.system, messages=self.messages,
                                        max_tokens=self._max_output_tokens,
                                        tools=self._tool_defs)
            self.usage.add_response(response)
            if response.stop_reason == "refusal":
                # Drop the whole turn so history stays valid (no empty assistant turn).
                del self.messages[turn_start:]
                return REFUSAL_REPLY
            # Append the full content, not just the text, so tool calls survive.
            self.messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use" or not self.tools:
                return response_text(response)
            self.messages.append({"role": "user", "content": self._run_tools(response)})
        return TOOL_LIMIT_REPLY

    def _run_tools(self, response) -> list[dict]:
        # All results go back in one user message, in call order.
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            outcome = self.tools.handle(block.name, dict(block.input or {}), self.turn)
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": outcome.content, "is_error": outcome.is_error})
        return results

    def _compact_if_needed(self) -> None:
        before = self.count_tokens()
        self.context_tokens = before
        if before <= self._context_limit:
            return
        result = self._compactor.compact(self.messages, self.id)
        if result is None:
            return
        self.messages = result.messages
        self.compaction_usage.add(result.usage)
        self.context_tokens = self.count_tokens()
        self.compactions.append(
            CompactionEvent(
                turn=self.turn,
                strategy=self._compactor.name,
                removed_messages=result.removed_messages,
                tokens_before=before,
                tokens_after=self.context_tokens,
            )
        )

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        total.add(self.usage)
        total.add(self.compaction_usage)
        return total
