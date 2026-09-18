"""Cost estimates from token usage.

Built-in prices are USD per million tokens for Claude on the Anthropic API (as of
2026-06). Any other model is priced only if the user sets a price on the
settings page; unpriced models are tracked in tokens, never guessed in dollars.
Cache reads bill at 0.1x the input price and 5-minute cache writes at 1.25x.
Estimates only: a refusal fallback bills at the fallback model's own rates, and
Bedrock bills at AWS rates (regional endpoints add 10%).
"""

from __future__ import annotations

from rewind.metrics import Usage

PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def price_for(model: str, custom: dict[str, tuple[float, float]] | None = None
              ) -> tuple[float, float] | None:
    """(input, output) USD per million tokens, or None when the model has no price."""
    if custom and model in custom:
        return custom[model]
    return PRICES.get(model.removeprefix("anthropic."))  # Bedrock model IDs


def estimate_cost(usage: Usage, model: str,
                  custom: dict[str, tuple[float, float]] | None = None) -> float | None:
    """Estimated USD cost of `usage`, or None when the model's price is unknown."""
    price = price_for(model, custom)
    if price is None:
        return None
    input_price, output_price = price
    return (
        usage.input_tokens * input_price
        + usage.cache_read_tokens * input_price * CACHE_READ_MULTIPLIER
        + usage.cache_write_tokens * input_price * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * output_price
    ) / 1_000_000
