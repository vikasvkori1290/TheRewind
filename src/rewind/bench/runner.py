"""Runs scenarios under each strategy and writes JSON + Markdown reports."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from rewind.bench.grading import grade, grade_control
from rewind.bench.scenarios import Scenario
from rewind.config import Settings
from rewind.factory import build_session
from rewind.llm import LLM
from rewind.pricing import estimate_cost
from rewind.store import ArchiveStore


@dataclass
class QuestionResult:
    kind: str
    question: str
    answer: str
    passed: bool


@dataclass
class RunResult:
    scenario: str
    strategy: str
    run: int
    seconds: float
    usage: dict
    cost_usd: float | None
    compactions: int
    recall: dict | None
    questions: list[QuestionResult] = field(default_factory=list)

    @property
    def graded(self) -> list[QuestionResult]:
        return [q for q in self.questions if q.kind == "question"]

    @property
    def correct(self) -> int:
        return sum(q.passed for q in self.graded)


Progress = Callable[[str], None]


@dataclass(frozen=True)
class Variant:
    """A strategy run under a label, optionally with its own context limit."""

    label: str
    strategy: str
    context_limit: int | None = None

    @classmethod
    def of(cls, strategy: str) -> Variant:
        return cls(strategy, strategy)


def run_scenario(scenario: Scenario, variant: Variant | str, settings: Settings, llm: LLM,
                 archive: ArchiveStore, run: int = 1, progress: Progress | None = None) -> RunResult:
    variant = Variant.of(variant) if isinstance(variant, str) else variant
    sid = f"{scenario.name}-{variant.strategy}-{run}-{uuid.uuid4().hex[:6]}"
    session = build_session(variant.strategy, settings, llm, archive=archive, session_id=sid,
                            context_limit=variant.context_limit)
    started = time.monotonic()
    results = []
    for i, turn in enumerate(scenario.turns, 1):
        answer = session.send(turn.text)
        if turn.kind == "question":
            results.append(QuestionResult("question", turn.text, answer, grade(answer, turn.expect)))
        elif turn.kind == "control":
            results.append(QuestionResult("control", turn.text, answer, grade_control(answer)))
        if progress:
            progress(f"{scenario.name}/{variant.label} run {run}: turn {i}/{len(scenario.turns)} "
                     f"context {session.context_tokens} tokens")
    usage = session.total_usage
    return RunResult(
        scenario=scenario.name, strategy=variant.label, run=run,
        seconds=round(time.monotonic() - started, 1),
        usage={**asdict(usage), "total_tokens": usage.total_tokens},
        cost_usd=estimate_cost(usage, settings.model),
        compactions=len(session.compactions),
        recall=session.tools.stats.to_dict() if session.tools else None,
        questions=results,
    )


def run_benchmark(scenarios: list[Scenario], variants: list[Variant | str], settings: Settings,
                  llm: LLM, archive: ArchiveStore, runs: int = 1,
                  progress: Progress | None = None) -> list[RunResult]:
    return [
        run_scenario(sc, variant, settings, llm, archive, run, progress)
        for run in range(1, runs + 1)
        for sc in scenarios
        for variant in variants
    ]


def summarize(results: list[RunResult]) -> list[dict]:
    rows = []
    for strategy in dict.fromkeys(r.strategy for r in results):
        rs = [r for r in results if r.strategy == strategy]
        questions = sum(len(r.graded) for r in rs)
        correct = sum(r.correct for r in rs)
        controls = [q for r in rs for q in r.questions if q.kind == "control"]
        tokens = sum(r.usage["total_tokens"] for r in rs)
        costs = [r.cost_usd for r in rs]
        rows.append({
            "strategy": strategy,
            "runs": len(rs),
            "accuracy": round(correct / questions, 3) if questions else None,
            "correct": correct,
            "questions": questions,
            "controls_passed": f"{sum(q.passed for q in controls)}/{len(controls)}",
            "input_tokens": sum(r.usage["input_tokens"] for r in rs),
            "output_tokens": sum(r.usage["output_tokens"] for r in rs),
            "cache_read_tokens": sum(r.usage["cache_read_tokens"] for r in rs),
            "total_tokens": tokens,
            "cost_usd": round(sum(costs), 4) if None not in costs else None,
            "tokens_per_correct": round(tokens / correct) if correct else None,
            "compactions": sum(r.compactions for r in rs),
            "recall_hits": sum((r.recall or {}).get("hits_by_id", 0)
                               + (r.recall or {}).get("hits_by_search", 0)
                               + (r.recall or {}).get("prefetches", 0) for r in rs),
            "model_calls": sum(r.usage["calls"] for r in rs),
        })
    return rows


def write_report(results: list[RunResult], out_dir: Path, model: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(results)
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(
        {"model": model, "summary": rows, "runs": [asdict(r) for r in results]}, indent=2))

    cols = ["strategy", "accuracy", "correct", "questions", "controls_passed", "input_tokens",
            "output_tokens", "cache_read_tokens", "total_tokens", "cost_usd",
            "tokens_per_correct", "compactions", "recall_hits", "model_calls"]
    lines = [f"# Rewind benchmark\n\nModel: `{model}`\n",
             "| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    lines += ["| " + " | ".join("" if row[c] is None else str(row[c]) for c in cols) + " |"
              for row in rows]
    lines.append("\nControl questions ask about things never discussed; they pass when the "
                 "answer says so (a keyword heuristic, so read the answers in results.json).")
    lines.append("\n## Answers\n")
    for r in results:
        lines.append(f"### {r.scenario} · {r.strategy} · run {r.run}\n")
        for q in r.questions:
            mark = "PASS" if q.passed else "FAIL"
            answer = q.answer.strip().replace("\n", " ")[:300]
            lines.append(f"- **{mark}** ({q.kind}) {q.question}\n  > {answer}")
        lines.append("")
    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(lines))
    return json_path, md_path
