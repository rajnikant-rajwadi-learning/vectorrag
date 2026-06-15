"""Token-aware, overlap-preserving text chunking.

We split on token counts (not characters) so each chunk fits the embedding model
cleanly and the retrieval context budget is predictable. Overlap preserves context
that would otherwise be cut mid-sentence across chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..tokens import _encoder
from .loader import LoadedPage


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def chunk_pages(
    pages: list[LoadedPage],
    *,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    embedding_model: str,
) -> list[Chunk]:
    """Chunk loaded pages into token-bounded, overlapping chunks.

    Sentences are packed greedily up to ``chunk_size_tokens``; when a chunk is
    flushed, the tail ``chunk_overlap_tokens`` worth of sentences seed the next
    one to preserve continuity.
    """
    enc = _encoder(embedding_model)
    chunks: list[Chunk] = []

    def tok_len(s: str) -> int:
        return len(enc.encode(s))

    for page in pages:
        sentences = _split_sentences(page.text) or [page.text]
        cur: list[str] = []
        cur_tokens = 0

        for sentence in sentences:
            st = tok_len(sentence)
            # A single oversized sentence: hard-split it on tokens.
            if st > chunk_size_tokens:
                if cur:
                    chunks.append(_make_chunk(cur, page, len(chunks)))
                    cur, cur_tokens = [], 0
                for piece in _hard_split(sentence, chunk_size_tokens, enc):
                    chunks.append(_make_chunk([piece], page, len(chunks)))
                continue

            if cur_tokens + st > chunk_size_tokens and cur:
                chunks.append(_make_chunk(cur, page, len(chunks)))
                cur, cur_tokens = _overlap_tail(cur, chunk_overlap_tokens, tok_len)

            cur.append(sentence)
            cur_tokens += st

        if cur:
            chunks.append(_make_chunk(cur, page, len(chunks)))

    return chunks


def _make_chunk(sentences: list[str], page: LoadedPage, idx: int) -> Chunk:
    text = " ".join(sentences).strip()
    return Chunk(
        id=f"{page.source}::p{page.page}::c{idx}",
        text=text,
        metadata={"source": page.source, "page": page.page, **page.metadata},
    )


def _overlap_tail(sentences: list[str], overlap_tokens: int, tok_len) -> tuple[list[str], int]:
    """Return the trailing sentences whose combined tokens ~= overlap_tokens."""
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        t = tok_len(sentence)
        if total + t > overlap_tokens:
            break
        tail.insert(0, sentence)
        total += t
    return tail, total


def _hard_split(text: str, size: int, enc) -> list[str]:
    ids = enc.encode(text)
    return [enc.decode(ids[i : i + size]) for i in range(0, len(ids), size)]
