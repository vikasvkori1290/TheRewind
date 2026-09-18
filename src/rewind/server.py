"""HTTP API and demo UI: one message goes to a plain and a Rewind session side by side.

    uv run rewind-serve            # http://127.0.0.1:8000

Local demo server: no authentication, bound to localhost by default, sessions
held in memory (the Rewind archive persists on disk).
"""

from __future__ import annotations

import argparse
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rewind.bench.scenarios import SCENARIOS
from rewind.config import Settings
from rewind.factory import build_session
from rewind.llm import LLM, AnthropicLLM
from rewind.pricing import estimate_cost
from rewind.session import Session
from rewind.store import ArchiveStore
from rewind.store_factory import open_archive

WEB_DIR = Path(__file__).parent / "web"
MAX_PAIRS = 20
MAX_MESSAGE_CHARS = 20_000


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class Pair:
    """A plain and a Rewind session that receive the same messages."""

    def __init__(self, plain: Session, rewind: Session):
        self.sessions = {"plain": plain, "rewind": rewind}
        self.lock = threading.Lock()
        self.seen = {"plain": (0, 0), "rewind": (0, 0)}  # compactions, recall events


def snapshot(session: Session, model: str) -> dict:
    usage = session.total_usage
    return {
        "strategy": session.strategy,
        "turn": session.turn,
        "context_tokens": session.context_tokens,
        "usage": {**asdict(usage), "total_tokens": usage.total_tokens},
        "cost_usd": estimate_cost(usage, model),
        "compactions": [asdict(c) for c in session.compactions],
        "recall": session.tools.stats.to_dict() if session.tools else None,
    }


def create_app(settings: Settings | None = None, llm: LLM | None = None,
               archive: ArchiveStore | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    archive = archive or open_archive(settings)
    get_llm: Callable[[], LLM] = (lambda: llm) if llm else (lambda: AnthropicLLM(settings))
    pairs: OrderedDict[str, Pair] = OrderedDict()
    pairs_lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=8)
    app = FastAPI(title="Rewind demo")

    def get_pair(pair_id: str) -> Pair:
        with pairs_lock:
            if pair_id not in pairs:
                raise HTTPException(404, "unknown pair")
            return pairs[pair_id]

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/config")
    def config():
        return {
            "model": settings.model,
            "context_limit": settings.context_limit,
            "scenarios": {
                name: [{"kind": t.kind, "text": t.text} for t in sc.turns]
                for name, sc in SCENARIOS.items()
            },
        }

    @app.post("/api/pairs")
    def create_pair():
        pair_id = uuid.uuid4().hex[:12]
        model_llm = get_llm()
        pair = Pair(build_session("plain", settings, model_llm, session_id=f"{pair_id}-plain"),
                    build_session("rewind", settings, model_llm, archive=archive,
                                  session_id=f"{pair_id}-rewind"))
        with pairs_lock:
            pairs[pair_id] = pair
            while len(pairs) > MAX_PAIRS:
                pairs.popitem(last=False)
        return {"id": pair_id}

    @app.get("/api/pairs/{pair_id}")
    def get_state(pair_id: str):
        pair = get_pair(pair_id)
        return {name: snapshot(s, settings.model) for name, s in pair.sessions.items()}

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
                except anthropic.APIError as e:
                    errors.append(f"{name}: model API error: {e.message}")
            if auth_error:
                raise HTTPException(401, f"Credentials rejected ({settings.provider}): "
                                         f"{auth_error.message}. Update the key in rewind/.env "
                                         "(see README, Providers) and restart the server.")
            if errors:
                raise HTTPException(502, "; ".join(errors))
            out = {}
            for name, reply in replies.items():
                session = pair.sessions[name]
                c_seen, r_seen = pair.seen[name]
                recalls = session.tools.events[r_seen:] if session.tools else []
                out[name] = {
                    "reply": reply,
                    "new_compactions": [asdict(c) for c in session.compactions[c_seen:]],
                    "new_recalls": [asdict(e) for e in recalls],
                    "state": snapshot(session, settings.model),
                }
                pair.seen[name] = (len(session.compactions), r_seen + len(recalls))
            return out
        finally:
            pair.lock.release()

    @app.get("/api/pairs/{pair_id}/archive")
    def list_archive(pair_id: str):
        pair = get_pair(pair_id)
        return [asdict(r) for r in archive.records(pair.sessions["rewind"].id)]

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Rewind demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
