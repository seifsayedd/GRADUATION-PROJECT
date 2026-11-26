"""FastAPI app. Build it with ``create_app``; run it with ``scripts/serve_phase6.py``."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..genai import rag
from ..genai.config import MissingSecretError, Settings
from ..genai.providers import RefusalError
from . import data
from .settings import ApiSettings

log = logging.getLogger("sentinelai.api")
ROOT = Path(__file__).resolve().parents[3]
MAX_ID = 120


class AskBody(BaseModel):
    question: str = Field(min_length=3, max_length=5000)


class _Lazy:
    """Builds a dependency on first use, so the read-only views start without loading models."""

    def __init__(self, factory: Callable[[], object]):
        self.factory, self.value = factory, None

    def get(self):
        if self.value is None:
            self.value = self.factory()
        return self.value


def default_store(root: Path):
    from ..genai.store import MemoryStore
    cache = root / "data" / "interim" / "phase5"
    if not (cache / "vectors.npy").exists():
        raise data.MissingOutput("no knowledge base yet: run `python scripts/genai_phase5.py ingest`")
    return MemoryStore.load(cache)


def create_app(root: Path = ROOT, settings: Settings | None = None, api: ApiSettings | None = None,
               store=None, embedder=None, provider=None, static_dir: Path | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    a = api or ApiSettings.from_env()

    def store_f():
        if store is not None:
            return store
        if s.database_url:
            from ..genai.store import PgVectorStore
            return PgVectorStore(s.database_url, s.embed_dim)
        return default_store(root)

    def embed_f():
        if embedder is not None:
            return embedder
        from ..genai.embeddings import get_embedder
        return get_embedder(s)

    def prov_f():
        if provider is not None:
            return provider
        from ..genai.providers import get_provider
        return get_provider(s)

    st, emb, prov = _Lazy(store_f), _Lazy(embed_f), _Lazy(prov_f)

    app = FastAPI(title="SentinelAI Blue Team API", version="0.6.0",
                  description="Read-only views over SentinelAI's real outputs, plus grounded reports and Q&A.")
    app.add_middleware(CORSMiddleware, allow_origins=list(a.cors_origins), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], allow_credentials=False)

    @app.exception_handler(data.MissingOutput)
    async def _missing(_: Request, exc: data.MissingOutput):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(MissingSecretError)
    async def _secret(_: Request, __: MissingSecretError):
        return JSONResponse(status_code=503, content={
            "detail": "ANTHROPIC_API_KEY is not set in the server's environment. Set it (see .env.example) "
                      "or use SENTINEL_LLM_PROVIDER=template."})

    @app.exception_handler(Exception)
    async def _any(_: Request, __: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "internal error; see the server log"})

    def explanations() -> dict[str, dict]:
        store_ = st.get()
        return {aid: store_.get_explanation(aid) for aid in store_.list_alerts()}

    def known(alert_id: str) -> dict:
        e = st.get().get_explanation(alert_id) if len(alert_id) <= MAX_ID else None
        if e is None:
            raise HTTPException(404, "unknown alert")
        return e

    @app.get("/api/health")
    def health():
        return {"status": "ok", "provider": s.provider, "embedder": s.embedder,
                "store": "postgres" if s.database_url else "memory", "auth": "none (local use only)"}

    @app.get("/api/summary")
    def summary():
        return data.summary(root)

    @app.get("/api/evaluation")
    def evaluation():
        return data.evaluation(root)

    @app.get("/api/alerts")
    def alerts(model: str | None = Query(None, max_length=80), flagged: bool | None = None,
               dataset: str | None = Query(None, max_length=40), stage: int | None = Query(None, ge=1, le=3)):
        rows = [data.alert_row(e) for e in explanations().values()]
        rows = [r for r in rows if (model is None or r["model"] == model)
                and (flagged is None or r["flagged"] == flagged)
                and (dataset is None or r["dataset"] == dataset) and (stage is None or r["stage"] == stage)]
        rows.sort(key=lambda r: (r["dataset"] or "", r["start_ms"] or 0, r["model"]))
        return {"count": len(rows), "rows": rows}

    @app.get("/api/alerts/{alert_id}")
    def alert(alert_id: str):
        return known(alert_id)

    def generate(fn):
        try:
            out = fn()
        except rag.UngroundedOutputError as e:
            raise HTTPException(422, {"rejected": True, "reason": "failed the groundedness check; not stored",
                                      "unsupported_numbers": e.result.unsupported_numbers,
                                      "unsupported_ips": e.result.unsupported_ips,
                                      "unknown_citations": e.result.unknown_citations})
        except RefusalError:
            raise HTTPException(422, {"rejected": True, "reason": "the model declined on safety grounds"})
        return {**out, "provider": prov.get().name}

    @app.post("/api/alerts/{alert_id}/report")
    def report(alert_id: str):
        known(alert_id)
        return generate(lambda: rag.report_for_alert(st.get(), emb.get(), prov.get(), alert_id, s.top_k))

    @app.post("/api/ask")
    def ask(body: AskBody):
        q = body.question.strip()
        if len(q) > a.max_question_chars:
            raise HTTPException(422, f"question is longer than {a.max_question_chars} characters")
        return generate(lambda: rag.ask(st.get(), emb.get(), prov.get(), q, s.top_k))

    @app.get("/api/sources/{source_id:path}")
    def source(source_id: str):
        """Text of a cited source, so the dashboard can show what a citation points to."""
        store_ = st.get()
        if source_id.startswith("alert:"):
            e = known(source_id[len("alert:"):])
            return {"id": source_id, "origin": "Phase 4 explanation object", "section": e["model"]["name"],
                    "text": e["summary"]}
        for c in getattr(store_, "chunks", []):
            if c.id == source_id:
                return {"id": c.id, "origin": c.source, "section": c.section, "text": c.text}
        raise HTTPException(404, "unknown source")

    @app.get("/api/timeline")
    def timeline():
        return data.timeline(root, explanations())

    @app.get("/api/attack-matrix")
    def attack_matrix():
        return data.attack_matrix(root, explanations())

    if static_dir is not None and static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="dashboard")
    return app
