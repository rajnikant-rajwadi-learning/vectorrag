"""Chroma vector store wrapper.

We persist locally on disk by default (``VECTORRAG_CHROMA_DIR``). In Lambda this
path should point at ``/tmp`` or, better, a directory hydrated from S3 / mounted
EFS (see DEPLOYMENT.md). We pass precomputed embeddings, so Chroma's own
embedding function is disabled — OpenAI is the single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from .logging_config import get_logger

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    score: float  # cosine similarity in [0,1]; higher is better


class VectorStore:
    def __init__(self, persist_dir: str, collection: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        # Cosine space matches OpenAI embeddings best.
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Insert or update chunks. Idempotent on id, so re-ingest is safe."""
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        log.info("upserted", count=len(ids), collection=self._collection.name)

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks above ``min_score``."""
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            if score >= min_score:
                out.append(RetrievedChunk(id=cid, text=doc, metadata=meta or {}, score=score))
        return out

    def count(self) -> int:
        return self._collection.count()
