"""End-to-end ingestion pipeline: files -> pages -> chunks -> embeddings -> Chroma."""

from __future__ import annotations

from pathlib import Path

from ..clients import build_embedder, build_openai_client, build_vector_store
from ..config import get_settings
from ..logging_config import get_logger
from .chunker import chunk_pages
from .loader import load_document

log = get_logger(__name__)

_SUPPORTED = {".pdf", ".html", ".htm", ".txt", ".md"}


def ingest_paths(paths: list[str | Path]) -> int:
    """Ingest one or more files/directories. Returns the number of chunks stored."""
    settings = get_settings()
    client = build_openai_client(settings)
    embedder = build_embedder(client, settings)
    store = build_vector_store(settings)

    files = _expand(paths)
    if not files:
        log.warning("no_supported_files", paths=[str(p) for p in paths])
        return 0

    total = 0
    for file in files:
        pages = load_document(file)
        chunks = chunk_pages(
            pages,
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            embedding_model=settings.embedding_model,
        )
        if not chunks:
            log.warning("empty_document", file=str(file))
            continue

        vectors = embedder.embed([c.text for c in chunks])
        store.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[c.metadata for c in chunks],
        )
        total += len(chunks)
        log.info("ingested_file", file=str(file), chunks=len(chunks))

    log.info("ingest_complete", files=len(files), chunks=total, total_in_store=store.count())
    return total


def _expand(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(f for f in sorted(p.rglob("*")) if f.suffix.lower() in _SUPPORTED)
        elif p.suffix.lower() in _SUPPORTED:
            files.append(p)
    return files
