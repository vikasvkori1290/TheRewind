"""A chat session that counts tokens and compacts when over the limit."""

from __future__ import annotations

from rewind.compaction import Compactor
from rewind.llm import LLM
from rewind.messages import response_text
from rewind.metrics import CompactionEvent, Usage

DEFAULT_SYSTEM = "You are a helpful assistant."
REFUSAL_REPLY = "[The model declined this request.]"


class Session:
    def __init__(
        self,
        session_id: str,
        llm: LLM,
        compactor: Compactor,
        context_limit: int,
        max_output_tokens: int,
        system: str = DEFAULT_SYSTEM,
    ):
        self.id = session_id
        self.system = system
        self.messages: list[dict] = []
        self.usage = Usage()
        self.compaction_usage = Usage()
        self.compactions: list[CompactionEvent] = []
        self.turn = 0
        self._llm = llm
        self._compactor = compactor
        self._context_limit = context_limit
        self._max_output_tokens = max_output_tokens

    def count_tokens(self) -> int:
        return self._llm.count_tokens(system=self.system, messages=self.messages)

    def send(self, text: str) -> str:
        self.turn += 1
        self.messages.append({"role": "user", "content": text})
        self._compact_if_needed()

        response = self._llm.create(
            system=self.system, messages=self.messages, max_tokens=self._max_output_tokens
        )
        self.usage.add_response(response)
        if response.stop_reason == "refusal":
            self.messages.pop()  # keep history valid: no empty assistant turn
            return REFUSAL_REPLY
        # Append the full content, not just the text, so tool calls survive.
        self.messages.append({"role": "assistant", "content": response.content})
        return response_text(response)

    def _compact_if_needed(self) -> None:
        before = self.count_tokens()
        if before <= self._context_limit:
            return
        result = self._compactor.compact(self.messages, self.id)
        if result is None:
            return
        self.messages = result.messages
        self.compaction_usage.add(result.usage)
        self.compactions.append(
            CompactionEvent(
                turn=self.turn,
                strategy=self._compactor.name,
                removed_messages=result.removed_messages,
                tokens_before=before,
                tokens_after=self.count_tokens(),
            )
        )

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        total.add(self.usage)
        total.add(self.compaction_usage)
        return total
