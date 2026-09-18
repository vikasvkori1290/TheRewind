import pytest
from fastapi.testclient import TestClient

from rewind.config import Settings
from rewind.server import MAX_MESSAGE_CHARS, create_app
from rewind.store import JsonlArchive

from fakes import FakeLLM


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(context_limit=120, keep_recent_messages=2), llm=FakeLLM(),
                     archive=JsonlArchive(tmp_path))
    return TestClient(app)


def test_index_and_config(client):
    assert "Rewind" in client.get("/").text
    cfg = client.get("/api/config").json()
    assert cfg["context_limit"] == 120
    assert "billing-service" in cfg["scenarios"]


def test_message_goes_to_both_sessions(client):
    pair = client.post("/api/pairs").json()["id"]
    out = client.post(f"/api/pairs/{pair}/messages", json={"text": "hello"}).json()
    assert set(out) == {"plain", "rewind"}
    assert out["plain"]["state"]["turn"] == 1 and out["rewind"]["state"]["turn"] == 1
    assert out["rewind"]["state"]["recall"] is not None
    assert out["plain"]["state"]["recall"] is None


def test_compaction_events_and_archive_listing(client):
    pair = client.post("/api/pairs").json()["id"]
    events = []
    for i in range(8):
        out = client.post(f"/api/pairs/{pair}/messages",
                          json={"text": f"message {i} " + "lorem ipsum " * 10}).json()
        events += out["rewind"]["new_compactions"]
    assert events
    records = client.get(f"/api/pairs/{pair}/archive").json()
    assert records and all(r["session"] == f"{pair}-rewind" for r in records)
    state = client.get(f"/api/pairs/{pair}").json()
    assert len(state["rewind"]["compactions"]) == len(events)


def test_validation_and_unknown_pair(client):
    pair = client.post("/api/pairs").json()["id"]
    assert client.post(f"/api/pairs/{pair}/messages", json={"text": ""}).status_code == 422
    too_long = "x" * (MAX_MESSAGE_CHARS + 1)
    assert client.post(f"/api/pairs/{pair}/messages", json={"text": too_long}).status_code == 422
    assert client.get("/api/pairs/nope").status_code == 404
