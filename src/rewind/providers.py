"""Registry of model providers Rewind can call.

Claude and Bedrock use Anthropic's SDK; every other provider speaks the OpenAI
Chat Completions protocol and goes through `openai_compat.py`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ANTHROPIC = "anthropic"
BEDROCK = "bedrock"
OPENAI_COMPAT = "openai_compat"


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    name: str
    kind: str  # anthropic | bedrock | openai_compat
    key_env: tuple[str, ...]  # environment variables used when no key is saved
    key_hint: str
    keys_url: str
    base_url: str | None = None
    default_model: str = ""
    # Model IDs to offer when the provider has no model-listing endpoint.
    suggested_models: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


PROVIDERS: dict[str, ProviderSpec] = {p.id: p for p in (
    ProviderSpec("anthropic", "Claude (Anthropic API)", ANTHROPIC, ("ANTHROPIC_API_KEY",),
                 "sk-ant-api…", "https://console.anthropic.com/settings/keys",
                 default_model="claude-opus-5"),
    ProviderSpec("bedrock", "Amazon Bedrock (Claude)", BEDROCK, ("AWS_BEARER_TOKEN_BEDROCK",),
                 "Bedrock API key (bedrock-api-key-…), or use AWS login",
                 "https://console.aws.amazon.com/bedrock/home#/api-keys",
                 default_model="claude-opus-5",
                 suggested_models=("claude-fable-5-1", "claude-fable-5", "claude-opus-5",
                                   "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5",
                                   "claude-haiku-4-5")),
    ProviderSpec("openai", "OpenAI (ChatGPT)", OPENAI_COMPAT, ("OPENAI_API_KEY",), "sk-…",
                 "https://platform.openai.com/api-keys", base_url="https://api.openai.com/v1"),
    ProviderSpec("gemini", "Google Gemini", OPENAI_COMPAT, ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                 "AIza…", "https://aistudio.google.com/apikey",
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai"),
    ProviderSpec("nim", "NVIDIA NIM", OPENAI_COMPAT, ("NVIDIA_API_KEY",), "nvapi-…",
                 "https://build.nvidia.com/", base_url="https://integrate.api.nvidia.com/v1",
                 default_model="openai/gpt-oss-20b"),
    ProviderSpec("openrouter", "OpenRouter", OPENAI_COMPAT, ("OPENROUTER_API_KEY",), "sk-or-…",
                 "https://openrouter.ai/keys", base_url="https://openrouter.ai/api/v1"),
    ProviderSpec("groq", "Groq", OPENAI_COMPAT, ("GROQ_API_KEY",), "gsk_…",
                 "https://console.groq.com/keys", base_url="https://api.groq.com/openai/v1"),
    ProviderSpec("mistral", "Mistral", OPENAI_COMPAT, ("MISTRAL_API_KEY",), "…",
                 "https://console.mistral.ai/api-keys", base_url="https://api.mistral.ai/v1"),
    ProviderSpec("deepseek", "DeepSeek", OPENAI_COMPAT, ("DEEPSEEK_API_KEY",), "sk-…",
                 "https://platform.deepseek.com/api_keys", base_url="https://api.deepseek.com/v1"),
    ProviderSpec("together", "Together AI", OPENAI_COMPAT, ("TOGETHER_API_KEY",), "…",
                 "https://api.together.ai/settings/api-keys", base_url="https://api.together.xyz/v1"),
    ProviderSpec("custom", "Custom (OpenAI-compatible)", OPENAI_COMPAT, (), "API key",
                 "", base_url=None),
)}


def get_provider(provider_id: str) -> ProviderSpec:
    if provider_id not in PROVIDERS:
        raise KeyError(f"unknown provider {provider_id!r}")
    return PROVIDERS[provider_id]


def usable_key(provider_id: str, key: str | None) -> bool:
    """Reject credentials that look valid but can't call the API."""
    if not key:
        return False
    # Claude subscription (Pro/Max) login tokens only work inside Claude's own apps.
    return not (provider_id == "anthropic" and key.startswith("sk-ant-oat"))
