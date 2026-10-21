"""REST API. Start with `cyberrag serve`, then open http://127.0.0.1:8000/docs.

Set CYBERRAG__API__KEY to require an `X-API-Key` header on every route except /health
and the docs pages."""

from __future__ import annotations

import json
import secrets
import threading
from collections.abc import Callable, Iterator
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from cyberrag import __version__
from cyberrag.config import Settings, get_settings
from cyberrag.errors import (
    ConfigError,
    CyberRAGError,
    EmbedderMismatchError,
    IndexMissingError,
    ProviderError,
)
from cyberrag.generation.rag import RAGPipeline, mark_cited, unsupported_ids
from cyberrag.retrieval.query import extract_cve_ids, extract_cwe_ids
from cyberrag.schemas import Answer, Hit, Message

PipelineFactory = Callable[[str], RAGPipeline]
_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_SECTION_ORDER = ["overview", "consequences", "mitigations", "detection", "examples", "observed"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    filters: dict[str, Any] | None = Field(
        default=None,
        description='e.g. {"severity": ["CRITICAL"], "known_exploited": true, '
                    '"published": {"gte": "2024-01-01"}}',
    )
    llm_profile: str | None = Field(default=None, description="A file name from config/llm/")
    history: list[Message] | None = None
    stream: bool = Field(default=False, description="Server-sent events: sources, token..., done")
    include_context: bool = False


def _status_for(exc: CyberRAGError) -> int:
    if isinstance(exc, ConfigError):
        return 400
    if isinstance(exc, IndexMissingError | EmbedderMismatchError):
        return 409
    if isinstance(exc, ProviderError):
        return 502
    return 500


def create_app(
    settings: Settings | None = None, pipeline_factory: PipelineFactory | None = None
) -> FastAPI:
    settings = settings or get_settings()
    factory = pipeline_factory or (lambda profile: RAGPipeline.from_settings(settings, profile))
    pipelines: dict[str, RAGPipeline] = {}
    lock = threading.Lock()

    def pipeline(profile: str | None = None) -> RAGPipeline:
        key = profile or settings.llm_profile
        with lock:
            if key not in pipelines:
                pipelines[key] = factory(key)
            return pipelines[key]

    expected_key = settings.api.key.get_secret_value().strip() if settings.api.key else ""

    def require_key(provided: str | None = Security(_KEY_HEADER)) -> None:
        if expected_key and not (provided and secrets.compare_digest(provided, expected_key)):
            raise HTTPException(401, "Missing or wrong X-API-Key header")

    protected = [Depends(require_key)]

    app = FastAPI(
        title="CyberRAG",
        version=__version__,
        description="Questions and answers over NVD CVE and MITRE CWE data.",
    )

    @app.exception_handler(CyberRAGError)
    def _handle(_: Request, exc: CyberRAGError) -> JSONResponse:
        return JSONResponse(status_code=_status_for(exc), content={"detail": str(exc)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        p = pipeline()
        retriever = p.retriever
        checks = {
            "llm": p.llm.health(),
            "embedder": retriever.embedder.health(),
            "vector_store": retriever.store.health(),
        }
        indexes = {}
        if checks["vector_store"][0]:
            indexes = {
                "cve": retriever.store.count(retriever.cve_collection),
                "cwe": retriever.store.count(retriever.cwe_collection),
            }
        return {
            "status": "ok" if all(ok for ok, _ in checks.values()) else "degraded",
            "checks": {k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()},
            "indexed_documents": indexes,
        }

    @app.get("/profiles", dependencies=protected)
    def profiles() -> dict[str, Any]:
        return {"default": settings.llm_profile, "profiles": settings.list_llm_profiles()}

    @app.post("/query", response_model=Answer, dependencies=protected)
    def query(req: QueryRequest) -> Any:
        p = pipeline(req.llm_profile)
        if not req.stream:
            return p.ask(req.question, req.filters, req.history, req.include_context)

        tokens, sources, prep = p.ask_stream(req.question, req.filters, req.history)

        def events() -> Iterator[str]:
            yield f"event: sources\ndata: {json.dumps([s.model_dump() for s in sources])}\n\n"
            parts: list[str] = []
            try:
                for token in tokens:
                    parts.append(token)
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
            except CyberRAGError as exc:
                yield f"event: error\ndata: {json.dumps(str(exc))}\n\n"
            answer = "".join(parts)
            done = {
                "cited": [s.ref for s in mark_cited(sources, answer) if s.cited],
                "unsupported": unsupported_ids(answer, prep.context),
            }
            yield f"event: done\ndata: {json.dumps(done)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/cve/{cve_id}", response_model=Hit, dependencies=protected)
    def get_cve(cve_id: str) -> Hit:
        ids = extract_cve_ids(cve_id)
        if not ids:
            raise HTTPException(400, "Expected an ID like CVE-2021-44228")
        retriever = pipeline().retriever
        hits = retriever.store.get(retriever.cve_collection, ids[:1])
        if hits:
            return hits[0]
        doc = retriever.live_lookup(ids[0]) if retriever.live_lookup else None
        if doc is None:
            raise HTTPException(404, f"{ids[0]} not found")
        return Hit(**doc.model_dump(), score=1.0, via="live")

    @app.get("/cwe/{cwe_id}", response_model=list[Hit], dependencies=protected)
    def get_cwe(cwe_id: str) -> list[Hit]:
        ids = extract_cwe_ids(cwe_id if cwe_id.upper().startswith("CWE") else f"CWE-{cwe_id}")
        if not ids:
            raise HTTPException(400, "Expected an ID like CWE-79")
        retriever = pipeline().retriever
        hits = retriever.store.find(retriever.cwe_collection, {"cwe_id": ids[0]}, limit=100)
        if not hits:
            raise HTTPException(404, f"{ids[0]} not found")
        order = {s: i for i, s in enumerate(_SECTION_ORDER)}
        return sorted(hits, key=lambda h: (order.get(h.payload.get("section"), 99),
                                           h.payload.get("part", 1)))

    return app
