"""Thin gateway around the Anthropic client.

Everything else depends on the `LLM` protocol, not on the SDK, so tests can use
a fake and the provider can be swapped without touching session or compaction.

Providers:
- anthropic: the Claude API (key in REWIND_API_KEY or ANTHROPIC_API_KEY).
- bedrock: Claude in Amazon Bedrock (Messages API endpoint). Authenticates with a
  Bedrock API key in AWS_BEARER_TOKEN_BEDROCK, or the usual AWS credential chain.
- nim: NVIDIA NIM or any OpenAI-compatible API, via `openai_compat.py`.
"""

from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import anthropic

from rewind.config import Settings
from rewind.messages import to_text

log = logging.getLogger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"
BEDROCK_PREFIX = "anthropic."


class LLM(Protocol):
    model: str

    def create(
        self, *, system: str, messages: list[dict], max_tokens: int, tools: list[dict] | None = None
    ) -> Any: ...

    def count_tokens(
        self, *, system: str, messages: list[dict], tools: list[dict] | None = None
    ) -> int: ...


def provider_model_id(settings: Settings) -> str:
    """Model ID as the provider expects it (Bedrock adds an `anthropic.` prefix)."""
    if settings.provider == "bedrock" and not settings.model.startswith(BEDROCK_PREFIX):
        return BEDROCK_PREFIX + settings.model
    return settings.model


def resolve_aws_region(settings: Settings) -> str | None:
    """REWIND_AWS_REGION, then AWS_REGION / AWS_DEFAULT_REGION, then the AWS config file."""
    region = settings.aws_region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if region:
        return region
    config_path = Path(os.getenv("AWS_CONFIG_FILE", Path.home() / ".aws" / "config"))
    if not config_path.is_file():
        return None
    parser = configparser.ConfigParser()
    parser.read(config_path)
    profile = os.getenv("AWS_PROFILE", "default")
    section = profile if profile == "default" else f"profile {profile}"
    return parser.get(section, "region", fallback=None)


def make_client(settings: Settings):
    if settings.provider == "anthropic":
        return anthropic.Anthropic(api_key=settings.api_key)
    if settings.provider == "bedrock":
        # A Bedrock API key comes from REWIND_API_KEY or AWS_BEARER_TOKEN_BEDROCK
        # (read by the SDK); REWIND_BEDROCK_AUTH=aws ignores keys and signs with
        # AWS credentials (aws login, profiles, roles) instead.
        kwargs: dict[str, Any] = {"aws_region": resolve_aws_region(settings)}
        if settings.bedrock_auth == "aws":
            os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            os.environ.pop("ANTHROPIC_AWS_API_KEY", None)
            if profile := os.getenv("AWS_PROFILE"):
                kwargs["aws_profile"] = profile
        elif settings.api_key:
            kwargs["api_key"] = settings.api_key
        return anthropic.AnthropicBedrockMantle(**kwargs)
    raise ValueError(f"unknown provider {settings.provider!r}; use 'anthropic' or 'bedrock'")


def make_llm(settings: Settings) -> LLM:
    """The model gateway for the configured provider."""
    if settings.provider == "nim":
        from rewind.openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(settings)
    return AnthropicLLM(settings)


def estimate_tokens(system: str, messages: list[dict], tools: list[dict] | None) -> int:
    """Rough count (about 4 characters per token) when token counting is unavailable."""
    chars = len(system) + sum(len(to_text(m)) for m in messages) + len(str(tools or ""))
    return chars // 4


class AnthropicLLM:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.model = provider_model_id(settings)
        # Server-side fallbacks exist only on the Claude API.
        self._fallbacks = settings.refusal_fallbacks and settings.provider == "anthropic"
        self._caching = settings.prompt_caching
        self._client = client or make_client(settings)
        self._can_count = True

    def create(self, *, system, messages, max_tokens, tools=None):
        kwargs: dict[str, Any] = dict(
            model=self.model, system=system, messages=messages, max_tokens=max_tokens
        )
        if tools:
            kwargs["tools"] = tools
        if self._caching:
            # Caches the prompt up to the last cacheable block. History is only
            # appended to between compactions, so each turn reads the previous
            # turn's prefix from cache; a compaction starts a new prefix.
            kwargs["cache_control"] = {"type": "ephemeral"}
        if self._fallbacks:
            # On a policy decline, the API re-runs the request on a fallback model.
            return self._client.beta.messages.create(
                betas=[_FALLBACK_BETA], fallbacks="default", **kwargs
            )
        return self._client.messages.create(**kwargs)

    def count_tokens(self, *, system, messages, tools=None):
        if self._can_count:
            kwargs: dict[str, Any] = dict(model=self.model, system=system, messages=messages)
            if tools:
                kwargs["tools"] = tools
            try:
                return self._client.messages.count_tokens(**kwargs).input_tokens
            except (anthropic.NotFoundError, anthropic.BadRequestError) as e:
                # Some endpoints lack token counting; estimate from then on.
                log.warning("token counting unavailable (%s); estimating instead", e.status_code)
                self._can_count = False
        return estimate_tokens(system, messages, tools)
