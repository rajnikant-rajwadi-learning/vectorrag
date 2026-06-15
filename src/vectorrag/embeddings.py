"""OpenAI embeddings client with batching and retry.

Wraps the OpenAI SDK so the rest of the app depends on a small, testable surface.
"""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .logging_config import get_logger

log = get_logger(__name__)

# OpenAI allows large batches, but keep it modest to bound request size/cost.
_BATCH_SIZE = 64


class Embedder:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching transparently."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        log.info("embedded", count=len(texts), model=self._model)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]
