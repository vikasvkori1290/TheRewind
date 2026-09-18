"""Benchmark command: `uv run rewind-bench --help`.

This calls the live API and costs money; it asks for confirmation unless --yes.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

from rewind.bench.runner import run_benchmark, summarize, write_report
from rewind.bench.scenarios import SCENARIOS
from rewind.config import Settings
from rewind.factory import STRATEGIES
from rewind.llm import make_llm
from rewind.store_factory import open_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare compaction strategies on planted facts")
    parser.add_argument("--strategies", default="plain,rewind",
                        help=f"comma list from {','.join(STRATEGIES)} (default plain,rewind)")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS),
                        help=f"comma list from {','.join(SCENARIOS)}")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--context-limit", type=int, default=None)
    parser.add_argument("--out", default="results")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if bad := [s for s in strategies if s not in STRATEGIES]:
        parser.error(f"unknown strategies: {bad}")
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if bad := [n for n in names if n not in SCENARIOS]:
        parser.error(f"unknown scenarios: {bad}")

    settings = Settings.from_env()
    if args.context_limit:
        settings = dataclasses.replace(settings, context_limit=args.context_limit)

    scenarios = [SCENARIOS[n] for n in names]
    turns = sum(len(s.turns) for s in scenarios) * len(strategies) * args.runs
    print(f"Model {settings.model} · limit {settings.context_limit} tokens · "
          f"{len(scenarios)} scenarios × {len(strategies)} strategies × {args.runs} runs "
          f"= {turns} chat turns (plus token counts, summaries and recalls).")
    if not args.yes and input("This uses the live API and costs money. Continue? [y/N] ") \
            .strip().lower() != "y":
        print("Cancelled.")
        return

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = open_archive(settings, out_dir / "archive")
    results = run_benchmark(scenarios, strategies, settings, make_llm(settings), archive,
                            runs=args.runs, progress=lambda msg: print(f"  {msg}", end="\r"))
    print()
    json_path, md_path = write_report(results, out_dir, settings.model)
    for row in summarize(results):
        print(f"{row['strategy']:>7}: accuracy {row['accuracy']} · tokens {row['total_tokens']} "
              f"· cost ${row['cost_usd']} · tokens/correct {row['tokens_per_correct']}")
    print(f"Report: {md_path}\nData:   {json_path}")


if __name__ == "__main__":
    main()
