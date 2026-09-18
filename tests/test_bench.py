import json
import re

from rewind.bench.grading import grade, grade_control, normalize
from rewind.bench.runner import run_benchmark, summarize, write_report
from rewind.bench.scenarios import SCENARIOS
from rewind.config import Settings
from rewind.messages import to_text
from rewind.search import query_terms
from rewind.store import JsonlArchive

from fakes import FakeLLM, fake_response, fake_tool_call


def test_normalize_and_grade():
    assert normalize("Limit: 1,237  req") == "limit: 1237 req"
    assert grade("It's 1,237 with a burst of 85.", [["1237"], ["85"]])
    assert not grade("It's 1,237.", [["1237"], ["85"]])
    assert grade("Row level security", [["row-level security", "row level security"]])


def test_grade_control():
    assert grade_control("I don't have any record of that.")
    assert not grade_control("We named it prod-east-7.")


def test_scenarios_are_well_formed():
    for sc in SCENARIOS.values():
        plants = "\n".join(t.text for t in sc.turns if t.kind == "plant")
        fillers = "\n".join(t.text for t in sc.turns if t.kind == "filler")
        for q in (t for t in sc.turns if t.kind == "question"):
            assert grade(plants, q.expect), f"{sc.name}: answer to {q.text!r} not planted"
            assert not grade(fillers, q.expect), f"{sc.name}: filler leaks {q.text!r}"
        # enough filler to force compaction at the default 8k limit (~4 chars/token)
        assert len(fillers) / 4 > Settings().context_limit


class OracleLLM(FakeLLM):
    """Answers questions the way a good model would: read pointers, recall, quote."""

    def create(self, *, system, messages, max_tokens, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        first = to_text(messages[0])
        if first.startswith("user: Summarize the conversation"):
            return fake_response("A summary without the details.")
        last = messages[-1]["content"]
        if isinstance(last, list) and last and isinstance(last[0], dict) \
                and last[0].get("type") == "tool_result":
            return fake_response(last[0]["content"])
        if isinstance(last, str) and not last.startswith(("Here are", "Please", "Note",
                                                         "Team", "Decision", "Keep", "Spec")):
            if tools and (rid := self._best_pointer(first, last)):
                return fake_tool_call({"id": rid, "level": "full"})
            return fake_response("I don't have that detail any more.")
        return fake_response("Noted.")

    @staticmethod
    def _best_pointer(header: str, question: str) -> str | None:
        terms = set(query_terms(question))
        best, best_score = None, 0
        for line in header.splitlines():
            if m := re.match(r"§([0-9a-f]{12}) → (.*)", line):
                score = len(terms & set(query_terms(m.group(2))))
                if score > best_score:
                    best, best_score = m.group(1), score
        return best


def test_benchmark_pipeline_and_report(tmp_path):
    settings = Settings(context_limit=8_000)
    results = run_benchmark([SCENARIOS["billing-service"]], ["none", "plain", "rewind"],
                            settings, OracleLLM(), JsonlArchive(tmp_path / "archive"))
    by = {r.strategy: r for r in results}
    assert by["none"].compactions == 0
    assert by["plain"].compactions >= 1 and by["rewind"].compactions >= 1
    assert by["rewind"].correct > by["plain"].correct
    assert by["rewind"].recall["hits_by_id"] >= 1

    json_path, md_path = write_report(results, tmp_path / "out", "fake-model")
    data = json.loads(json_path.read_text())
    assert {row["strategy"] for row in data["summary"]} == {"none", "plain", "rewind"}
    assert "| strategy |" in md_path.read_text()
    rows = {r["strategy"]: r for r in summarize(results)}
    assert rows["rewind"]["accuracy"] > rows["plain"]["accuracy"]
