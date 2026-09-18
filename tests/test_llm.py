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


def test_bedrock_prefixes_model_and_skips_fallbacks():
    c, plain, beta = client()
    llm = AnthropicLLM(Settings(provider="bedrock"), c)
    llm.create(system="s", messages=[], max_tokens=10)
    assert llm.model == "anthropic.claude-opus-5"
    call = plain.calls[0]
    assert call["model"] == "anthropic.claude-opus-5" and "fallbacks" not in call
    assert call["cache_control"] == {"type": "ephemeral"}
    assert beta.calls == []


def test_count_tokens_falls_back_to_estimate_when_unsupported():
    import anthropic
    import httpx

    class NoCount(Recorder):
        def count_tokens(self, **kwargs):
            self.calls.append(kwargs)
            req = httpx.Request("POST", "https://x/v1/messages/count_tokens")
            raise anthropic.NotFoundError("nope", response=httpx.Response(404, request=req), body=None)

    messages = NoCount()
    c = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=Recorder()))
    llm = AnthropicLLM(Settings(provider="bedrock"), c)
    msgs = [{"role": "user", "content": "x" * 400}]
    assert llm.count_tokens(system="", messages=msgs) >= 100
    llm.count_tokens(system="", messages=msgs)
    assert len(messages.calls) == 1  # stops asking after the first failure


def test_bedrock_cost_uses_base_model_price():
    from rewind.metrics import Usage
    from rewind.pricing import estimate_cost

    u = Usage(input_tokens=1_000_000)
    assert estimate_cost(u, "anthropic.claude-opus-5") == estimate_cost(u, "claude-opus-5") == 5.0


def test_region_resolution_order(monkeypatch, tmp_path):
    from rewind.llm import resolve_aws_region

    cfg = tmp_path / "config"
    cfg.write_text("[default]\nregion = eu-west-1\n[profile work]\nregion = ap-south-1\n")
    for var in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg))
    assert resolve_aws_region(Settings()) == "eu-west-1"
    monkeypatch.setenv("AWS_PROFILE", "work")
    assert resolve_aws_region(Settings()) == "ap-south-1"
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert resolve_aws_region(Settings()) == "us-west-2"
    assert resolve_aws_region(Settings(aws_region="us-east-2")) == "us-east-2"
