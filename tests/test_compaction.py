import re

from rewind.compaction import NullCompactor, PlainCompactor, RewindCompactor
from rewind.messages import HEADER_PREFIX, make_gist, split_turns, strip_header
from rewind.store import JsonlArchive

from fakes import FakeLLM, FakeTextBlock

CODE = "def compute_late_fee(days_overdue, daily_rate=0.75):\n    return min(days_overdue * daily_rate, 30.0)"


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": [FakeTextBlock(text)]}


def history(n):
    msgs = [user(f"Keep this function:\n{CODE}"), assistant("Saved.")]
    for i in range(1, n):
        msgs += [user(f"question {i}"), assistant(f"answer {i}")]
    return msgs


def test_null_compactor_never_compacts():
    assert NullCompactor().compact(history(5), "s") is None


def test_rewind_archives_every_old_turn_verbatim(tmp_path):
    archive = JsonlArchive(tmp_path)
    comp = RewindCompactor(FakeLLM(), archive, keep_recent=2, summary_max_tokens=100)
    msgs = history(4)
    result = comp.compact(msgs, "s")

    assert result.removed_messages == 6
    assert len(result.archived_ids) == 3
    texts = [archive.text(i) for i in result.archived_ids]
    assert CODE in texts[0]
    assert "question 2" in texts[2] and "answer 2" in texts[2]


def test_every_pointer_resolves_to_a_record(tmp_path):
    archive = JsonlArchive(tmp_path)
    comp = RewindCompactor(FakeLLM(), archive, keep_recent=2, summary_max_tokens=100)
    result = comp.compact(history(4), "s")
    header = result.messages[0]["content"]
    ids = re.findall(r"§([0-9a-f]{12})", header)
    assert ids and all(archive.get(i) for i in ids)
    assert "compute_late_fee" in header  # gist names the defined function


def test_repeated_compaction_keeps_old_pointers_and_does_not_archive_headers(tmp_path):
    archive = JsonlArchive(tmp_path)
    comp = RewindCompactor(FakeLLM(), archive, keep_recent=2, summary_max_tokens=100)
    first = comp.compact(history(4), "s")
    msgs = first.messages + [assistant("a"), user("next 1"), assistant("b"), user("next 2"),
                             assistant("c")]
    second = comp.compact(msgs, "s")

    header = second.messages[0]["content"]
    for rid in first.archived_ids:
        assert f"§{rid}" in header
    for rid in second.archived_ids:
        assert HEADER_PREFIX not in archive.text(rid)


def test_pointer_table_is_capped(tmp_path):
    archive = JsonlArchive(tmp_path)
    comp = RewindCompactor(FakeLLM(), archive, keep_recent=2, summary_max_tokens=100,
                           max_pointers=2)
    header = comp.compact(history(6), "s").messages[0]["content"]
    assert header.count("§") == 2
    assert "older archived turns" in header


def test_plain_and_rewind_keep_the_same_recent_messages(tmp_path):
    msgs = history(4)
    plain = PlainCompactor(FakeLLM(), keep_recent=2, summary_max_tokens=100).compact(msgs, "s")
    rewind = RewindCompactor(FakeLLM(), JsonlArchive(tmp_path), keep_recent=2,
                             summary_max_tokens=100).compact(msgs, "s")
    assert plain.messages[1:] == rewind.messages[1:]
    assert strip_header(plain.messages[0]) == strip_header(rewind.messages[0])


def test_split_turns_and_gist():
    turns = split_turns(history(3))
    assert len(turns) == 3
    assert make_gist(turns[0]) == "Keep this function: [compute_late_fee]"
    assert make_gist(turns[1]) == "question 1"
