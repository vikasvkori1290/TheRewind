"""HTTP API and pages: the side-by-side demo (/) and provider settings (/settings).

    uv run rewind-serve            # http://127.0.0.1:8000

Local server for one user: no login, so it only answers requests addressed to
localhost and rejects cross-site requests (another website can't change keys or
spend credits). API keys are stored on disk (mode 0600) and only ever returned
masked. Chat sessions live in memory; the archive and usage ledger persist.
"""

from __future__ import annotations

import argparse
import dataclasses
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from rewind.appconfig import AppConfig
from rewind.bench.scenarios import SCENARIOS
from rewind.config import Settings
from rewind.factory import build_session
from rewind.ledger import MeteredLLM, UsageLedger
from rewind.llm import LLM, list_models, make_llm
from rewind.pricing import PRICES, estimate_cost, price_for
from rewind.providers import get_provider
from rewind.session import Session
from rewind.store import ArchiveStore
from rewind.store_factory import open_archive

WEB_DIR = Path(__file__).parent / "web"
MAX_PAIRS = 20
MAX_MESSAGE_CHARS = 20_000
LOCAL_HOSTS = ("127.0.0.1", "localhost")


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class ProviderIn(BaseModel):
    api_key: str | None = Field(default=None, max_length=4096)
    region: str | None = Field(default=None, max_length=40)
    auth: str | None = Field(default=None, max_length=10)
    base_url: str | None = Field(default=None, max_length=500)


class ActiveIn(BaseModel):
    provider: str = Field(max_length=40)
    model: str = Field(max_length=200)


class PriceIn(BaseModel):
    model: str = Field(max_length=200)
    input_per_mtok: float
    output_per_mtok: float


class Pair:
    """A plain and a Rewind session that receive the same messages."""

    def __init__(self, plain: Session, rewind: Session, provider: str, model: str):
        self.sessions = {"plain": plain, "rewind": rewind}
        self.provider, self.model = provider, model
        self.lock = threading.Lock()
        self.seen = {"plain": (0, 0), "rewind": (0, 0)}  # compactions, recall events


def snapshot(session: Session, model: str, prices: dict) -> dict:
    usage = session.total_usage
    return {
        "strategy": session.strategy,
        "turn": session.turn,
        "context_tokens": session.context_tokens,
        "usage": {**asdict(usage), "total_tokens": usage.total_tokens},
        "cost_usd": estimate_cost(usage, model, prices),
        "compactions": [asdict(c) for c in session.compactions],
        "recall": session.tools.stats.to_dict() if session.tools else None,
    }


def _error_message(e: Exception) -> str:
    return getattr(e, "message", None) or str(e) or type(e).__name__


def create_app(settings: Settings | None = None, llm: LLM | None = None,
               archive: ArchiveStore | None = None, config: AppConfig | None = None,
               ledger: UsageLedger | None = None,
               allowed_hosts: tuple[str, ...] = LOCAL_HOSTS) -> FastAPI:
    env = settings or Settings.from_env()
    archive = archive or open_archive(env)
    config = config or AppConfig(env.data_dir, env)
    ledger = ledger or UsageLedger(env.data_dir)
    pairs: OrderedDict[str, Pair] = OrderedDict()
    pairs_lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=8)
    app = FastAPI(title="Rewind")

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # Only answer requests addressed to this machine (blocks DNS rebinding) and,
        # for changes, only JSON from our own pages (blocks cross-site requests).
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in allowed_hosts:
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).hostname not in allowed_hosts:
                return JSONResponse({"detail": "cross-site request refused"}, status_code=403)
            if request.method in ("POST", "PUT") and \
                    not request.headers.get("content-type", "").startswith("application/json"):
                return JSONResponse({"detail": "JSON required"}, status_code=415)
        return await call_next(request)

    def current_llm() -> tuple[LLM, Settings]:
        s = config.settings()
        inner = llm or make_llm(s)
        return MeteredLLM(inner, ledger, s.provider, config.custom_prices), s

    def get_pair(pair_id: str) -> Pair:
        with pairs_lock:
            if pair_id not in pairs:
                raise HTTPException(404, "unknown pair")
            return pairs[pair_id]

    def bad_request(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e).strip("'\"")) from None

    # --- pages ----------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/settings", include_in_schema=False)
    def settings_page():
        return FileResponse(WEB_DIR / "settings.html")

    # --- demo -----------------------------------------------------------------

    @app.get("/api/config")
    def app_config():
        active = config.active()
        return {
            "provider": active["provider"],
            "provider_name": get_provider(active["provider"]).name,
            "model": active["model"],
            "priced": price_for(active["model"], config.custom_prices()) is not None,
            "context_limit": env.context_limit,
            "scenarios": {
                name: [{"kind": t.kind, "text": t.text} for t in sc.turns]
                for name, sc in SCENARIOS.items()
            },
        }

    @app.post("/api/pairs")
    def create_pair():
        try:
            model_llm, s = current_llm()
        except Exception as e:  # missing key, bad base URL, missing optional package
            raise HTTPException(400, f"Can't use the active model: {_error_message(e)}. "
                                     "Check the settings page.") from None
        pair_id = uuid.uuid4().hex[:12]
        pair = Pair(build_session("plain", s, model_llm, session_id=f"{pair_id}-plain"),
                    build_session("rewind", s, model_llm, archive=archive,
                                  session_id=f"{pair_id}-rewind"),
                    s.provider, s.model)
        with pairs_lock:
            pairs[pair_id] = pair
            while len(pairs) > MAX_PAIRS:
                pairs.popitem(last=False)
        return {"id": pair_id, "provider": s.provider, "model": s.model}

    @app.get("/api/pairs/{pair_id}")
    def get_state(pair_id: str):
        pair = get_pair(pair_id)
        prices = config.custom_prices()
        return {name: snapshot(s, pair.model, prices) for name, s in pair.sessions.items()}

    @app.post("/api/pairs/{pair_id}/messages")
    def send(pair_id: str, body: MessageIn):
        pair = get_pair(pair_id)
        if not pair.lock.acquire(blocking=False):
            raise HTTPException(409, "a message is already being processed")
        try:
            futures = {name: pool.submit(s.send, body.text) for name, s in pair.sessions.items()}
            # Wait for both before answering, so the lock covers all session work.
            replies, errors, auth_error = {}, [], None
            for name, future in futures.items():
                try:
                    replies[name] = future.result()
                except anthropic.AuthenticationError as e:
                    auth_error = e
                except Exception as e:  # provider errors from either SDK
                    if getattr(e, "status_code", None) == 401:
                        auth_error = e
                    else:
                        errors.append(f"{name}: model API error: {_error_message(e)}")
            if auth_error:
                raise HTTPException(401, f"Credentials rejected ({pair.provider}): "
                                         f"{_error_message(auth_error)}. Update the key on the "
                                         "settings page.")
            if errors:
                raise HTTPException(502, "; ".join(errors))
            prices = config.custom_prices()
            out = {}
            for name, reply in replies.items():
                session = pair.sessions[name]
                c_seen, r_seen = pair.seen[name]
                recalls = session.tools.events[r_seen:] if session.tools else []
                out[name] = {
                    "reply": reply,
                    "new_compactions": [asdict(c) for c in session.compactions[c_seen:]],
                    "new_recalls": [asdict(e) for e in recalls],
                    "state": snapshot(session, pair.model, prices),
                }
                pair.seen[name] = (len(session.compactions), r_seen + len(recalls))
            return out
        finally:
            pair.lock.release()

    @app.get("/api/pairs/{pair_id}/archive")
    def list_archive(pair_id: str):
        pair = get_pair(pair_id)
        return [asdict(r) for r in archive.records(pair.sessions["rewind"].id)]

    # --- settings: providers, active model, prices, usage ------------------------

    @app.get("/api/providers")
    def providers():
        return {"providers": config.provider_status(), "active": config.active()}

    @app.put("/api/providers/{provider_id}")
    def update_provider(provider_id: str, body: ProviderIn):
        bad_request(config.set_provider, provider_id, **body.model_dump(exclude_none=True))
        return {"ok": True}

    @app.delete("/api/providers/{provider_id}/key")
    def remove_key(provider_id: str):
        bad_request(config.remove_key, provider_id)
        return {"ok": True}

    @app.post("/api/providers/{provider_id}/test")
    def test_provider(provider_id: str):
        bad_request(get_provider, provider_id)
        entry = next(p for p in config.provider_status() if p["id"] == provider_id)
        key, _ = config.key_for(provider_id)
        base = config.settings()
        s = dataclasses.replace(base, provider=provider_id, api_key=key,
                                base_url=entry["base_url"],
                                aws_region=entry["region"] or base.aws_region,
                                bedrock_auth=entry["auth"])
        try:
            models = list_models(s)
        except Exception as e:
            return {"ok": False, "error": _error_message(e)[:300]}
        note = ("Bedrock has no model list here; these are the Claude models it serves. "
                "Send a message to confirm the key." if entry["kind"] == "bedrock" else None)
        return {"ok": True, "models": models[:500], "count": len(models), "note": note}

    @app.put("/api/active")
    def set_active(body: ActiveIn):
        bad_request(config.set_active, body.provider, body.model.strip())
        return {"ok": True, "active": config.active()}

    @app.get("/api/prices")
    def prices():
        return {"builtin": {m: list(p) for m, p in PRICES.items()},
                "custom": {m: list(p) for m, p in config.custom_prices().items()}}

    @app.put("/api/prices")
    def set_price(body: PriceIn):
        bad_request(config.set_price, body.model.strip(), body.input_per_mtok,
                    body.output_per_mtok)
        return {"ok": True}

    @app.delete("/api/prices/{model:path}")
    def remove_price(model: str):
        config.remove_price(model)
        return {"ok": True}

    @app.get("/api/usage")
    def usage():
        return ledger.summary()

    @app.delete("/api/usage")
    def reset_usage():
        ledger.reset()
        return {"ok": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Rewind demo server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    # Always bound to localhost: the API manages secrets and has no login.
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
