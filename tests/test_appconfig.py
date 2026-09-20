import json
import stat

import pytest

from rewind.appconfig import AppConfig, mask
from rewind.config import Settings
from rewind.ledger import MeteredLLM, UsageLedger
from rewind.metrics import Usage
from rewind.pricing import estimate_cost

# pyrefly: ignore [missing-import]
from fakes import FakeLLM, fake_response

ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY",
            "GOOGLE_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "REWIND_API_KEY")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return AppConfig(tmp_path, Settings(provider="nim", model="openai/gpt-oss-20b"))


def test_saved_key_is_private_and_masked(cfg, tmp_path):
    cfg.set_provider("openai", api_key="sk-test-1234567890abcd")
    path = tmp_path / "rewind_config.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    status = {p["id"]: p for p in cfg.provider_status()}
    assert status["openai"]["key_masked"] == "••••abcd"
    assert "sk-test" not in json.dumps(status)
    assert cfg.key_for("openai") == ("sk-test-1234567890abcd", "saved")


def test_env_fallback_and_removal(cfg, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-env-9999")
    assert cfg.key_for("nim") == ("nvapi-from-env-9999", "env")
    cfg.set_provider("nim", api_key="nvapi-saved-0000")
    assert cfg.key_for("nim")[1] == "saved"
    cfg.remove_key("nim")
    assert cfg.key_for("nim")[1] == "env"


def test_subscription_tokens_are_rejected(cfg, monkeypatch):
    with pytest.raises(ValueError, match="subscription"):
        cfg.set_provider("anthropic", api_key="sk-ant-oat01-abc")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-oat01-abc")
    assert cfg.key_for("anthropic") == (None, None)


@pytest.mark.parametrize("kwargs", [
    {"api_key": "has space"}, {"api_key": ""}, {"region": "moon-1"}, {"auth": "password"},
    {"base_url": "http://evil.example.com/v1"}])
def test_invalid_values_are_rejected(cfg, kwargs):
    provider = "custom" if "base_url" in kwargs else "bedrock"
    with pytest.raises(ValueError):
        cfg.set_provider(provider, **kwargs)


def test_unknown_provider(cfg):
    with pytest.raises(KeyError):
        cfg.set_provider("nope", api_key="x")


def test_active_settings_carry_key_and_options(cfg):
    cfg.set_provider("bedrock", api_key="bedrock-api-key-abc", region="eu-west-1", auth="key")
    cfg.set_active("bedrock", "claude-sonnet-5")
    s = cfg.settings()
    assert (s.provider, s.model, s.api_key, s.aws_region) == (
        "bedrock", "claude-sonnet-5", "bedrock-api-key-abc", "eu-west-1")
    with pytest.raises(ValueError):
        cfg.set_active("openai", "bad model id!")


def test_custom_base_url(cfg):
    cfg.set_provider("custom", api_key="k-123456789", base_url="http://localhost:11434/v1/")
    cfg.set_active("custom", "llama3")
    assert cfg.settings().base_url == "http://localhost:11434/v1"


def test_prices(cfg):
    cfg.set_price("gpt-test", 2.0, 8.0)
    assert cfg.custom_prices() == {"gpt-test": (2.0, 8.0)}
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost(u, "gpt-test", cfg.custom_prices()) == 10.0
    assert estimate_cost(u, "unpriced-model", cfg.custom_prices()) is None
    with pytest.raises(ValueError):
        cfg.set_price("gpt-test", -1, 2)
    cfg.remove_price("gpt-test")
    assert cfg.custom_prices() == {}


def test_config_survives_reload(cfg, tmp_path):
    cfg.set_provider("groq", api_key="gsk_abcdefghij")
    cfg.set_active("groq", "llama-x")
    again = AppConfig(tmp_path, Settings())
    assert again.key_for("groq")[0] == "gsk_abcdefghij"
    assert again.active() == {"provider": "groq", "model": "llama-x"}


def test_ledger_tracks_dollars_only_for_priced_models(tmp_path):
    ledger = UsageLedger(tmp_path)
    priced = MeteredLLM(FakeLLM(model="claude-opus-5", replies=[fake_response("a", 1000, 100)]),
                        ledger, "anthropic")
    free = MeteredLLM(FakeLLM(model="openai/gpt-oss-20b", replies=[fake_response("b", 500, 50)]),
                      ledger, "nim")
    msgs = [{"role": "user", "content": "hi"}]
    priced.create(system="", messages=msgs, max_tokens=10)
    free.create(system="", messages=msgs, max_tokens=10)
    s = ledger.summary()
    rows = {r["model"]: r for r in s["rows"]}
    assert rows["claude-opus-5"]["cost_usd"] == pytest.approx((1000 * 5 + 100 * 25) / 1e6)
    assert rows["openai/gpt-oss-20b"]["cost_usd"] is None
    assert s["unpriced_tokens"] == 550 and s["calls"] == 2
    assert s["total_cost_usd"] == pytest.approx(0.0075)
    ledger.reset()
    assert ledger.summary()["calls"] == 0


def test_mask():
    assert mask("sk-1234567890") == "••••7890"
    assert mask("short") == "••••"
