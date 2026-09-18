"""Thin gateway around the Anthropic client.

Everything else depends on the `LLM` protocol, not on the SDK, so tests can use
a fake and the provider can be swapped without touching session or compaction.
"""

from __future__ import annotations

from typing import Any, Protocol

import anthropic

from rewind.config import Settings

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLM(Protocol):
    model: str

    def create(
        self, *, system: str, messages: list[dict], max_tokens: int, tools: list[dict] | None = None
    ) -> Any: ...

    def count_tokens(
        self, *, system: str, messages: list[dict], tools: list[dict] | None = None
    ) -> int: ...


class AnthropicLLM:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.model = settings.model
        self._fallbacks = settings.refusal_fallbacks
        self._caching = settings.prompt_caching
        self._client = client or anthropic.Anthropic()

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
        kwargs: dict[str, Any] = dict(model=self.model, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
        return self._client.messages.count_tokens(**kwargs).input_tokens
