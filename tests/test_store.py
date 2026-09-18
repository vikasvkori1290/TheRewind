import json

import pytest

from rewind.store import JsonlArchive, content_id

STORES = {"jsonl": JsonlArchive}


@pytest.fixture(params=sorted(STORES))
def make_store(request, tmp_path):
    cls = STORES[request.param]
    return lambda: cls(tmp_path)


def test_put_is_idempotent(make_store, tmp_path):
    store = make_store()
    a = store.put("same text", session="s1", gist="g")
    b = store.put("same text", session="s1", gist="other gist")
    assert a.id == b.id
    assert len(store.records("s1")) == 1


def test_text_round_trips_exactly(make_store):
    store = make_store()
    text = "def f():\n    return 'ünïcode' \t\n"
    rec = store.put(text, session="s1", gist="f")
    assert store.text(rec.id) == text
    assert rec.chars == len(text)


def test_same_text_in_two_sessions_stays_isolated(make_store):
    store = make_store()
    a = store.put("shared text", session="s1", gist="g")
    b = store.put("shared text", session="s2", gist="g")
    assert a.id != b.id
    assert [r.id for r in store.records("s1")] == [a.id]


def test_search_never_crosses_sessions(make_store):
    store = make_store()
    store.put("the release manager is Priya Raman", session="s1", gist="people")
    assert store.search("release manager", session="s2") == []
    hits = store.search("release manager", session="s1")
    assert hits and hits[0].coverage == 1.0


def test_search_ranks_exact_identifier_first(make_store):
    store = make_store()
    store.put("we discussed the weather and lunch plans", session="s", gist="chat")
    target = store.put("def compute_late_fee(days, rate=0.75): ...", session="s", gist="code")
    store.put("the late train was delayed by fees", session="s", gist="other")
    hits = store.search("compute_late_fee", session="s")
    assert hits[0].record.id == target.id


def test_search_finds_numbers_written_with_commas(make_store):
    store = make_store()
    rec = store.put("Our API rate limit is 1,237 requests per minute.", session="s", gist="limit")
    hits = store.search("1237 requests", session="s")
    assert hits[0].record.id == rec.id


def test_search_with_only_stopwords_returns_nothing(make_store):
    store = make_store()
    store.put("anything at all", session="s", gist="g")
    assert store.search("what is the", session="s") == []


def test_reload_from_disk(make_store):
    store = make_store()
    rec = store.put("persist me", session="s", gist="p", kind="code", turns=[3, 4])
    reloaded = make_store()
    got = reloaded.get(rec.id)
    assert got == rec
    assert reloaded.text(rec.id) == "persist me"
    assert reloaded.search("persist", session="s")[0].record.id == rec.id


@pytest.mark.parametrize("bad", ["../../etc/passwd", "ABCDEF123456", "abc", "", "0123456789abX"])
def test_invalid_ids_are_rejected(make_store, bad):
    store = make_store()
    assert store.get(bad) is None
    assert store.text(bad) is None


def test_jsonl_log_is_one_line_per_record(tmp_path):
    store = JsonlArchive(tmp_path)
    store.put("one", session="s", gist="1")
    store.put("two", session="s", gist="2")
    store.put("one", session="s", gist="1")
    lines = (tmp_path / "archive.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == content_id("s", "one")
