"""Usage ledger: every model call, with tokens and (for priced models) USD cost.

`MeteredLLM` wraps any LLM and records each call, so chat turns, compaction
summaries and recall round trips are all counted, whatever the provider.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rewind.metrics import Usage
from rewind.pricing import estimate_cost

LEDGER_FILE = "usage.jsonl"


@dataclass(frozen=True)
class UsageEntry:
    time: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None  # None: the model has no price, so only tokens are tracked


class UsageLedger:
    def __init__(self, data_dir: str | Path):
        self._path = Path(data_dir) / LEDGER_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, provider: str, model: str, usage: Usage, cost_usd: float | None) -> None:
        entry = UsageEntry(
            time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            provider=provider, model=model, input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens, cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens, cost_usd=cost_usd)
        with self._lock, self._path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def entries(self) -> list[UsageEntry]:
        if not self._path.is_file():
            return []
        return [UsageEntry(**json.loads(line))
                for line in self._path.read_text().splitlines() if line.strip()]

    def summary(self) -> dict:
        rows: dict[tuple[str, str], dict] = {}
        for e in self.entries():
            row = rows.setdefault((e.provider, e.model), {
                "provider": e.provider, "model": e.model, "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0,
                "cost_usd": None, "priced_calls": 0})
            row["calls"] += 1
            for field in ("input_tokens", "output_tokens", "cache_read_tokens",
                          "cache_write_tokens"):
                row[field] += getattr(e, field)
            if e.cost_usd is not None:
                row["cost_usd"] = (row["cost_usd"] or 0.0) + e.cost_usd
                row["priced_calls"] += 1
        items = sorted(rows.values(), key=lambda r: (-(r["cost_usd"] or 0), -r["calls"]))
        for r in items:
            r["total_tokens"] = (r["input_tokens"] + r["output_tokens"]
                                 + r["cache_read_tokens"] + r["cache_write_tokens"])
            if r["cost_usd"] is not None:
                r["cost_usd"] = round(r["cost_usd"], 6)
        return {
            "rows": items,
            "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in items), 6),
            "total_tokens": sum(r["total_tokens"] for r in items),
            "unpriced_tokens": sum(r["total_tokens"] for r in items if r["cost_usd"] is None),
            "calls": sum(r["calls"] for r in items),
        }

    def reset(self) -> None:
        with self._lock:
            self._path.unlink(missing_ok=True)


class MeteredLLM:
    """Wraps an LLM and records every call in the ledger."""

    def __init__(self, inner: Any, ledger: UsageLedger, provider: str,
                 prices: Callable[[], dict[str, tuple[float, float]]] = dict):
        self._inner = inner
        self._ledger = ledger
        self._provider = provider
        self._prices = prices
        self.model = inner.model

    def create(self, **kwargs):
        response = self._inner.create(**kwargs)
        usage = Usage()
        usage.add_response(response)
        # Price by the configured model name (Bedrock IDs are normalized in pricing).
        self._ledger.record(self._provider, self.model, usage,
                            estimate_cost(usage, self.model, self._prices()))
        return response

    def count_tokens(self, **kwargs) -> int:
        return self._inner.count_tokens(**kwargs)
