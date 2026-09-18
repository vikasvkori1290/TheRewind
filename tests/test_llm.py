from types import SimpleNamespace

from rewind.config import Settings
from rewind.llm import AnthropicLLM


class Recorder:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "ok"

    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(input_tokens=42)


def client():
    messages, beta = Recorder(), Recorder()
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=beta)), messages, beta


def test_default_uses_fallbacks_and_caching():
    c, plain, beta = client()
    AnthropicLLM(Settings(), c).create(system="s", messages=[], max_tokens=10)
    call = beta.calls[0]
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["model"] == "claude-opus-5" and "tools" not in call
    assert plain.calls == []


def test_fallbacks_and_caching_can_be_disabled():
    c, plain, beta = client()
    llm = AnthropicLLM(Settings(refusal_fallbacks=False, prompt_caching=False), c)
    llm.create(system="s", messages=[], max_tokens=10, tools=[{"name": "recall"}])
    call = plain.calls[0]
    assert "cache_control" not in call and "fallbacks" not in call
    assert call["tools"] == [{"name": "recall"}]
    assert beta.calls == []


def test_count_tokens_passes_tools():
    c, plain, _ = client()
    n = AnthropicLLM(Settings(), c).count_tokens(system="s", messages=[], tools=[{"name": "t"}])
    assert n == 42 and plain.calls[0]["tools"] == [{"name": "t"}]


def test_rewind_api_key_is_used_and_hidden(monkeypatch, tmp_path):
    from rewind.config import load_dotenv

    env = tmp_path / ".env"
    env.write_text("# comment\nREWIND_API_KEY='sk-ant-api-test'\nREWIND_MODEL=\n")
    monkeypatch.delenv("REWIND_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    s = Settings.from_env()
    assert s.api_key == "sk-ant-api-test"
    assert "sk-ant" not in repr(s)
    assert s.model == "claude-opus-5"  # empty values are ignored
    monkeypatch.setenv("REWIND_API_KEY", "from-shell")
    load_dotenv(env)
    import os
    assert os.environ["REWIND_API_KEY"] == "from-shell"  # shell wins over .env
