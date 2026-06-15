"""Retrieval: embed the query, search Chroma, enforce a token budget.

Returns citable snippets ([S1], [S2], ...) and only as many as fit within
``max_context_tokens`` so the final prompt stays inside the model's window.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder
from .security import neutralize_context
from .tokens import count_tokens
from .vectorstore import VectorStore


@dataclass
class Snippet:
    id: str  # citable id, e.g. "S1"
    text: str
    metadata: dict
    score: float


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, settings: Settings):
        self._embedder = embedder
        self._store = store
        self._settings = settings

    def retrieve(self, query: str) -> list[Snippet]:
        query_vec = self._embedder.embed_query(query)
        hits = self._store.query(
            query_embedding=query_vec,
            top_k=self._settings.top_k,
            min_score=self._settings.min_relevance_score,
        )

        snippets: list[Snippet] = []
        used_tokens = 0
        for i, hit in enumerate(hits, start=1):
            text = neutralize_context(hit.text)
            t = count_tokens(text, self._settings.chat_model)
            if used_tokens + t > self._settings.max_context_tokens:
                break
            snippets.append(
                Snippet(id=f"S{i}", text=text, metadata=hit.metadata, score=hit.score)
            )
            used_tokens += t
        return snippets
