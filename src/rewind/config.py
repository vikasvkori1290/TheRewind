"""Runtime settings, read from environment variables with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into the environment without overriding existing variables."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() not in ("0", "false", "no", "")


@dataclass(frozen=True)
class Settings:
    model: str = "claude-opus-5"
    # Deliberately small so compaction happens within a short demo conversation.
    context_limit: int = 8_000
    # Latest messages that compaction never touches.
    keep_recent_messages: int = 4
    max_output_tokens: int = 16_000
    summary_max_tokens: int = 2_000
    # Share of meaningful query terms a search hit must contain (0-1).
    recall_min_coverage: float = 0.5
    # Server-side refusal fallback (Claude API only; disable on Bedrock/Vertex/Foundry).
    refusal_fallbacks: bool = True
    # Automatic prompt caching of the conversation prefix.
    prompt_caching: bool = True
    store: str = "jsonl"  # jsonl | sqlite
    data_dir: str = "data"
    # Rewind's own key, so it never collides with ANTHROPIC_API_KEY used by other
    # tools. When empty, the SDK falls back to its usual credential lookup.
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        d = cls()
        return cls(
            model=os.getenv("REWIND_MODEL", d.model),
            context_limit=int(os.getenv("REWIND_CONTEXT_LIMIT", d.context_limit)),
            keep_recent_messages=int(
                os.getenv("REWIND_KEEP_RECENT_MESSAGES", d.keep_recent_messages)
            ),
            max_output_tokens=int(os.getenv("REWIND_MAX_OUTPUT_TOKENS", d.max_output_tokens)),
            summary_max_tokens=int(os.getenv("REWIND_SUMMARY_MAX_TOKENS", d.summary_max_tokens)),
            recall_min_coverage=float(
                os.getenv("REWIND_RECALL_MIN_COVERAGE", d.recall_min_coverage)
            ),
            refusal_fallbacks=_flag("REWIND_REFUSAL_FALLBACKS", d.refusal_fallbacks),
            prompt_caching=_flag("REWIND_PROMPT_CACHING", d.prompt_caching),
            store=os.getenv("REWIND_STORE", d.store),
            data_dir=os.getenv("REWIND_DATA_DIR", d.data_dir),
            api_key=os.getenv("REWIND_API_KEY") or None,
        )
