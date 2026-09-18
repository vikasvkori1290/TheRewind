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


def _api_key(provider: str) -> str | None:
    # REWIND_API_KEY wins; OpenAI-compatible providers also accept their own variables
    # (NVIDIA_API_KEY, OPENAI_API_KEY, ...). Claude and Bedrock let the SDK find its own.
    from rewind.providers import OPENAI_COMPAT, PROVIDERS

    key = os.getenv("REWIND_API_KEY")
    spec = PROVIDERS.get(provider)
    if not key and spec and spec.kind == OPENAI_COMPAT:
        key = next((v for var in spec.key_env if (v := os.getenv(var))), None)
    return key or None


@dataclass(frozen=True)
class Settings:
    model: str = "claude-opus-5"
    # Deliberately small so compaction happens within a short demo conversation.
    context_limit: int = 8_000
    # Latest messages that compaction never touches.
    keep_recent_messages: int = 4
    max_output_tokens: int = 16_000
    summary_max_tokens: int = 2_000
    # Pointers listed in the compaction header; older turns stay searchable.
    max_pointers: int = 15
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
    provider: str = "anthropic"  # anthropic | bedrock | nim (any OpenAI-compatible API)
    base_url: str | None = None  # nim only; defaults to NVIDIA's hosted endpoint
    request_timeout: float = 600.0
    aws_region: str | None = None  # bedrock only; falls back to AWS_REGION / AWS config
    bedrock_auth: str = "key"  # key (Bedrock API key) | aws (AWS credentials, SigV4)

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
            max_pointers=int(os.getenv("REWIND_MAX_POINTERS", d.max_pointers)),
            recall_min_coverage=float(
                os.getenv("REWIND_RECALL_MIN_COVERAGE", d.recall_min_coverage)
            ),
            refusal_fallbacks=_flag("REWIND_REFUSAL_FALLBACKS", d.refusal_fallbacks),
            prompt_caching=_flag("REWIND_PROMPT_CACHING", d.prompt_caching),
            store=os.getenv("REWIND_STORE", d.store),
            data_dir=os.getenv("REWIND_DATA_DIR", d.data_dir),
            api_key=_api_key(os.getenv("REWIND_PROVIDER", d.provider)),
            provider=os.getenv("REWIND_PROVIDER", d.provider),
            base_url=os.getenv("REWIND_BASE_URL") or None,
            request_timeout=float(os.getenv("REWIND_REQUEST_TIMEOUT", d.request_timeout)),
            aws_region=os.getenv("REWIND_AWS_REGION") or None,
            bedrock_auth=os.getenv("REWIND_BEDROCK_AUTH", d.bedrock_auth),
        )
