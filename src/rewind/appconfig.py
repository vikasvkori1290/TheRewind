"""Local, private configuration edited from the settings page.

Holds provider API keys, provider options (Bedrock region/auth, custom base
URL), the active provider and model, and user-set model prices. Stored as JSON
readable only by the current user (mode 0600) in the data directory, which git
ignores. Full keys never leave this module except to build an API client; the
API returns masked values only.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import tempfile
import threading
from pathlib import Path

from rewind.config import Settings
from rewind.providers import BEDROCK, PROVIDERS, get_provider, usable_key

CONFIG_FILE = "rewind_config.json"
MAX_KEY_CHARS = 4096
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
REGION = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d$")


def mask(key: str) -> str:
    return "••••" + key[-4:] if len(key) > 8 else "••••"


def validate_base_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not (url.startswith("https://") or re.match(r"^http://(localhost|127\.0\.0\.1)(:\d+)?(/|$)", url)):
        raise ValueError("base URL must use https (or http://localhost)")
    return url


class AppConfig:
    def __init__(self, data_dir: str | Path, env_settings: Settings | None = None):
        self._path = Path(data_dir) / CONFIG_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._env = env_settings or Settings.from_env()
        self._lock = threading.Lock()
        self._data = self._load()

    # --- persistence -------------------------------------------------------

    def _load(self) -> dict:
        if self._path.is_file():
            data = json.loads(self._path.read_text())
        else:
            data = {}
        data.setdefault("providers", {})
        data.setdefault("prices", {})
        data.setdefault("active", None)
        return data

    def _save(self) -> None:
        # Write to a private temp file, then atomically replace the config.
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".rewind_config.")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # --- keys and provider options -----------------------------------------

    def _entry(self, provider_id: str) -> dict:
        get_provider(provider_id)
        return self._data["providers"].setdefault(provider_id, {})

    def key_for(self, provider_id: str) -> tuple[str | None, str | None]:
        """(key, source) where source is 'saved', 'env' or None."""
        saved = self._data["providers"].get(provider_id, {}).get("api_key")
        if usable_key(provider_id, saved):
            return saved, "saved"
        # The provider chosen in .env / the shell also accepts REWIND_API_KEY.
        if provider_id == self._env.provider and usable_key(provider_id, self._env.api_key):
            return self._env.api_key, "env"
        for var in get_provider(provider_id).key_env:
            if usable_key(provider_id, value := os.getenv(var)):
                return value, "env"
        return None, None

    def set_provider(self, provider_id: str, *, api_key: str | None = None,
                     region: str | None = None, auth: str | None = None,
                     base_url: str | None = None) -> None:
        with self._lock:
            entry = self._entry(provider_id)
            if api_key is not None:
                api_key = api_key.strip()
                if not api_key or len(api_key) > MAX_KEY_CHARS or any(c.isspace() for c in api_key):
                    raise ValueError("API key must be non-empty, without spaces")
                if not usable_key(provider_id, api_key):
                    raise ValueError("Claude Pro/Max subscription tokens (sk-ant-oat…) can't call "
                                     "the API; create an API key in the Anthropic Console")
                entry["api_key"] = api_key
            if region is not None:
                if region and not REGION.match(region):
                    raise ValueError("region must look like us-east-1")
                entry["region"] = region or None
            if auth is not None:
                if auth not in ("key", "aws"):
                    raise ValueError("auth must be 'key' or 'aws'")
                entry["auth"] = auth
            if base_url is not None:
                entry["base_url"] = validate_base_url(base_url) if base_url else None
            self._save()

    def remove_key(self, provider_id: str) -> None:
        with self._lock:
            self._entry(provider_id).pop("api_key", None)
            self._save()

    def provider_status(self) -> list[dict]:
        out = []
        for spec in PROVIDERS.values():
            key, source = self.key_for(spec.id)
            entry = self._data["providers"].get(spec.id, {})
            ready = bool(key) or (spec.kind == BEDROCK and entry.get("auth") == "aws")
            if spec.id == "custom":
                ready = ready and bool(entry.get("base_url"))
            out.append({
                **spec.to_dict(),
                "key_set": bool(key), "key_source": source,
                "key_masked": mask(key) if key else None,
                "region": entry.get("region"), "auth": entry.get("auth", "key"),
                "base_url": entry.get("base_url") or spec.base_url,
                "ready": ready,
            })
        return out

    # --- active provider and model -------------------------------------------

    def active(self) -> dict:
        if self._data["active"]:
            return dict(self._data["active"])
        return {"provider": self._env.provider, "model": self._env.model}

    def set_active(self, provider_id: str, model: str) -> None:
        get_provider(provider_id)
        if not MODEL_ID.match(model):
            raise ValueError("invalid model id")
        with self._lock:
            self._data["active"] = {"provider": provider_id, "model": model}
            self._save()

    def settings(self) -> Settings:
        """Settings for the active provider and model, with its key and options."""
        active = self.active()
        provider_id = active["provider"]
        spec = get_provider(provider_id)
        entry = self._data["providers"].get(provider_id, {})
        key, _ = self.key_for(provider_id)
        return dataclasses.replace(
            self._env,
            provider=provider_id,
            model=active["model"],
            api_key=key,
            base_url=entry.get("base_url") or spec.base_url,
            aws_region=entry.get("region") or self._env.aws_region,
            bedrock_auth=entry.get("auth") or self._env.bedrock_auth,
        )

    # --- prices --------------------------------------------------------------

    def custom_prices(self) -> dict[str, tuple[float, float]]:
        return {m: (p[0], p[1]) for m, p in self._data["prices"].items()}

    def set_price(self, model: str, input_per_mtok: float, output_per_mtok: float) -> None:
        if not MODEL_ID.match(model):
            raise ValueError("invalid model id")
        if not (0 <= input_per_mtok <= 1000 and 0 <= output_per_mtok <= 1000):
            raise ValueError("prices must be between 0 and 1000 USD per million tokens")
        with self._lock:
            self._data["prices"][model] = [input_per_mtok, output_per_mtok]
            self._save()

    def remove_price(self, model: str) -> None:
        with self._lock:
            self._data["prices"].pop(model, None)
            self._save()


def runtime(env: Settings | None = None):
    """(settings, metered LLM, config) for command-line tools, matching the settings page."""
    from rewind.ledger import MeteredLLM, UsageLedger
    from rewind.llm import make_llm

    env = env or Settings.from_env()
    config = AppConfig(env.data_dir, env)
    settings = config.settings()
    llm = MeteredLLM(make_llm(settings), UsageLedger(env.data_dir), settings.provider,
                     config.custom_prices)
    return settings, llm, config
