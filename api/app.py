"""FastAPI application exposing the RAG engine over HTTP.

Runs the same engine locally (``uvicorn api.app:app``) and in AWS Lambda (via the
Mangum adapter in ``lambda_handler.py``). The engine is built lazily and reused
across requests / warm invocations.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from vectorrag.config import get_settings
from vectorrag.logging_config import configure_logging, get_logger
from vectorrag.rag import RAGEngine
from vectorrag.security import SecurityError

configure_logging(os.getenv("LOG_LEVEL", "INFO"))
log = get_logger("api")

app = FastAPI(title="VectorRAG 10-Q API", version="0.1.0")

_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    """Lazily construct and cache the engine (one per warm container)."""
    global _engine
    if _engine is None:
        _engine = RAGEngine(get_settings())
    return _engine


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class SourceModel(BaseModel):
    id: str
    source: str
    page: int | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceModel]
    grounded: bool
    latency_ms: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, engine: Annotated[RAGEngine, Depends(get_engine)]) -> AskResponse:
    try:
        resp = engine.answer(req.question)
    except SecurityError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.error("ask_failed", error=str(type(exc).__name__))
        raise HTTPException(status_code=502, detail="Upstream model error") from exc

    return AskResponse(
        answer=resp.answer,
        sources=[SourceModel(**vars(s)) for s in resp.sources],
        grounded=resp.grounded,
        latency_ms=resp.latency_ms,
    )
