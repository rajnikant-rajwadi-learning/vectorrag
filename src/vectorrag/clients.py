"""Factory helpers that wire together the OpenAI client and store.

Keeping construction in one place keeps the RAG engine and CLI/API thin.
"""

from __future__ import annotations

from openai import OpenAI

from .config import Settings, get_settings
from .embeddings import Embedder
from .vectorstore import VectorStore


def build_openai_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or get_settings()
    return OpenAI(api_key=settings.resolve_api_key(), timeout=60.0, max_retries=0)


def build_embedder(client: OpenAI, settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    return Embedder(client=client, model=settings.embedding_model)


def build_vector_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    return VectorStore(persist_dir=settings.chroma_dir, collection=settings.collection)
