"""Terminal chat for trying a session by hand.

    uv run rewind-chat               # Rewind (archive + recall)
    uv run rewind-chat --plain       # baseline: plain compaction
"""

from __future__ import annotations

import argparse

from rewind.appconfig import runtime
from rewind.factory import build_session
from rewind.pricing import estimate_cost
from rewind.store_factory import open_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a Rewind session")
    parser.add_argument("--plain", action="store_true", help="use plain compaction instead")
    args = parser.parse_args()

    settings, llm, config = runtime()
    strategy = "plain" if args.plain else "rewind"
    session = build_session(strategy, settings, llm, archive=open_archive(settings))
    print(f"Rewind chat · {strategy} · {settings.provider} · {settings.model} · "
          f"limit {settings.context_limit} tokens · session {session.id}")
    print("Commands: /stats, /archive, /quit\n")

    seen_compactions = seen_recalls = 0
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
            cost = estimate_cost(u, settings.model, config.custom_prices())
            cost_text = f"${cost:.4f}" if cost is not None else "unknown model price"
            print(f"  context {session.context_tokens} tokens · calls {u.calls} · "
                  f"input {u.input_tokens} · output {u.output_tokens} · "
                  f"cache read {u.cache_read_tokens} · cost ≈ {cost_text}")
            if session.tools:
                print(f"  recall {session.tools.stats.to_dict()}")
            continue
        if text == "/archive":
            if session.tools:
                for r in session.tools.archive.records(session.id):
                    print(f"  §{r.id} [{r.kind}] {r.gist}")
            continue

        print(f"claude> {session.send(text)}\n")
        for event in session.compactions[seen_compactions:]:
            print(f"  -- compacted: removed {event.removed_messages} messages, "
                  f"{event.tokens_before} -> {event.tokens_after} tokens --")
        seen_compactions = len(session.compactions)
        if session.tools:
            for e in session.tools.events[seen_recalls:]:
                print(f"  -- recall {e.outcome}: {e.id or e.query!r} ({e.level}) --")
            seen_recalls = len(session.tools.events)


if __name__ == "__main__":
    main()
