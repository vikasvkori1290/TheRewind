"""Terminal chat for trying a session by hand: `uv run rewind-chat`."""

from __future__ import annotations

import uuid

from rewind.compaction import PlainCompactor
from rewind.config import Settings
from rewind.llm import AnthropicLLM
from rewind.session import Session


def build_session(settings: Settings) -> Session:
    llm = AnthropicLLM(settings)
    compactor = PlainCompactor(llm, settings.keep_recent_messages, settings.summary_max_tokens)
    return Session(
        session_id=uuid.uuid4().hex[:8],
        llm=llm,
        compactor=compactor,
        context_limit=settings.context_limit,
        max_output_tokens=settings.max_output_tokens,
    )


def main() -> None:
    settings = Settings.from_env()
    session = build_session(settings)
    print(f"Rewind chat · model {settings.model} · limit {settings.context_limit} tokens")
    print("Type /stats for token usage, /quit to exit.\n")
    seen_compactions = 0
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/stats":
            u = session.total_usage
            print(f"  calls {u.calls} · input {u.input_tokens} · output {u.output_tokens}"
                  f" · compactions {len(session.compactions)}")
            continue
        print(f"claude> {session.send(text)}\n")
        for event in session.compactions[seen_compactions:]:
            print(f"  -- compacted: removed {event.removed_messages} messages, "
                  f"{event.tokens_before} -> {event.tokens_after} tokens --\n")
        seen_compactions = len(session.compactions)


if __name__ == "__main__":
    main()
