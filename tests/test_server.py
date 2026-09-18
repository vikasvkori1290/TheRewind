import pytest
from fastapi.testclient import TestClient

from rewind.config import Settings
from rewind.server import MAX_MESSAGE_CHARS, create_app
from rewind.store import JsonlArchive

from fakes import FakeLLM


ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY", "REWIND_API_KEY",
            "AWS_BEARER_TOKEN_BEDROCK", "GEMINI_API_KEY", "GOOGLE_API_KEY")


@pytest.fixture
def client(tmp_path, monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    settings = Settings(context_limit=120, keep_recent_messages=2, data_dir=str(tmp_path),
                        provider="anthropic", model="claude-opus-5")
    app = create_app(settings, llm=FakeLLM(model="claude-opus-5"),
                     archive=JsonlArchive(tmp_path / "archive"), allowed_hosts=("testserver",))
    return TestClient(app)


def new_pair(client):
    return client.post("/api/pairs", json={}).json()["id"]


def test_index_and_config(client):
    assert "Rewind" in client.get("/").text
    assert "Provider keys" in client.get("/settings").text
    cfg = client.get("/api/config").json()
    assert cfg["context_limit"] == 120
    assert "billing-service" in cfg["scenarios"]


def test_message_goes_to_both_sessions(client):
    pair = new_pair(client)
    out = client.post(f"/api/pairs/{pair}/messages", json={"text": "hello"}).json()
    assert set(out) == {"plain", "rewind"}
    assert out["plain"]["state"]["turn"] == 1 and out["rewind"]["state"]["turn"] == 1
    assert out["rewind"]["state"]["recall"] is not None
    assert out["plain"]["state"]["recall"] is None


def test_compaction_events_and_archive_listing(client):
    pair = new_pair(client)
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
    pair = new_pair(client)
    assert client.post(f"/api/pairs/{pair}/messages", json={"text": ""}).status_code == 422
    too_long = "x" * (MAX_MESSAGE_CHARS + 1)
    assert client.post(f"/api/pairs/{pair}/messages", json={"text": too_long}).status_code == 422
    assert client.get("/api/pairs/nope").status_code == 404


def test_usage_ledger_records_calls_with_cost(client):
    pair = new_pair(client)
    client.post(f"/api/pairs/{pair}/messages", json={"text": "hello"})
    usage = client.get("/api/usage").json()
    assert usage["calls"] == 2  # one call per pane
    assert usage["rows"][0]["model"] == "claude-opus-5" and usage["total_cost_usd"] > 0


def test_keys_are_never_returned_in_full(client):
    r = client.put("/api/providers/openai", json={"api_key": "sk-secret-abcdef123456"})
    assert r.status_code == 200
    body = client.get("/api/providers").text
    assert "sk-secret" not in body and "••••3456" in body
    assert client.put("/api/providers/nope", json={"api_key": "x"}).status_code == 400
    assert client.put("/api/providers/anthropic",
                      json={"api_key": "sk-ant-oat01-x"}).status_code == 400


def test_active_model_and_prices(client):
    assert client.put("/api/active", json={"provider": "openai", "model": "gpt-x"}).json()["ok"]
    cfg = client.get("/api/config").json()
    assert cfg["provider"] == "openai" and cfg["priced"] is False
    client.put("/api/prices", json={"model": "gpt-x", "input_per_mtok": 1, "output_per_mtok": 4})
    assert client.get("/api/config").json()["priced"] is True
    assert "gpt-x" in client.get("/api/prices").json()["custom"]
    client.delete("/api/prices/gpt-x")
    assert client.get("/api/prices").json()["custom"] == {}


def test_cross_site_and_foreign_host_requests_are_refused(client):
    r = client.put("/api/providers/openai", json={"api_key": "sk-abcdefghijkl"},
                   headers={"Origin": "https://evil.example.com"})
    assert r.status_code == 403
    r = client.put("/api/providers/openai", content='{"api_key": "sk-abcdefghijkl"}',
                   headers={"Content-Type": "text/plain"})
    assert r.status_code == 415
    assert client.get("/api/providers", headers={"Host": "attacker.example"}).status_code == 403


def test_provider_test_reports_errors_without_crashing(client):
    r = client.post("/api/providers/custom/test", json={}).json()
    assert r["ok"] is False and "base URL" in r["error"]
    r = client.post("/api/providers/bedrock/test", json={}).json()
    assert r["ok"] and "claude-opus-5" in r["models"]
