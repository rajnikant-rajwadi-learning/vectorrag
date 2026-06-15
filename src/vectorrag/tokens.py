"""Token counting helpers built on tiktoken.

Centralising token math lets us enforce a strict context budget so we never blow
past the model's context window or rack up surprise cost.
"""

from __future__ import annotations

import functools

import tiktoken


@functools.lru_cache(maxsize=8)
def _encoder(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Newer models may not be in tiktoken's registry yet; o200k_base is the
        # encoding used by gpt-4o / text-embedding-3-*.
        return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Return the number of tokens in ``text`` for ``model``."""
    if not text:
        return 0
    return len(_encoder(model).encode(text))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    """Hard-truncate ``text`` to at most ``max_tokens`` tokens."""
    enc = _encoder(model)
    ids = enc.encode(text)
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])
