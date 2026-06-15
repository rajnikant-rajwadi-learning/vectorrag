"""Central configuration.

All tunables are read from environment variables (prefixed ``VECTORRAG_``) or a
local ``.env`` file. In AWS the OpenAI key is pulled from Secrets Manager so it
never lives on disk or in an environment variable baked into an image.
"""

from __future__ import annotations

import functools
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_prefix="VECTORRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    # OPENAI_API_KEY has no prefix (OpenAI SDK convention), so it is read directly.
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Generation controls (anti-hallucination) ---
    temperature: float = 0.0
    max_output_tokens: int = 800

    # --- Retrieval / token budget ---
    top_k: int = 5
    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 80
    max_context_tokens: int = 6000
    # Chroma returns cosine *distance* (0 = identical). We convert to a
    # similarity score in [0,1] and drop anything below this floor.
    min_relevance_score: float = 0.20

    # --- Vector store ---
    chroma_dir: str = "./.chroma"
    collection: str = "tenq_filings"

    # --- Conversation memory ---
    max_history_tokens: int = 2000

    # --- AWS ---
    openai_secret_name: str = ""
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _check_overlap(cls, v: int, info) -> int:
        size = info.data.get("chunk_size_tokens", 500)
        if v >= size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return v

    def resolve_api_key(self) -> str:
        """Return the OpenAI API key, fetching from Secrets Manager when configured.

        Resolution order:
        1. ``VECTORRAG_OPENAI_SECRET_NAME`` -> AWS Secrets Manager (production / Lambda)
        2. ``OPENAI_API_KEY`` environment variable (local dev)
        """
        if self.openai_secret_name:
            return _fetch_secret(self.openai_secret_name, self.aws_region)
        if self.openai_api_key:
            return self.openai_api_key
        raise RuntimeError(
            "No OpenAI API key found. Set OPENAI_API_KEY locally or "
            "VECTORRAG_OPENAI_SECRET_NAME in AWS."
        )


@functools.lru_cache(maxsize=4)
def _fetch_secret(secret_name: str, region: str) -> str:
    """Fetch and cache a secret string from AWS Secrets Manager.

    Cached so that warm Lambda invocations do not re-hit Secrets Manager.
    Supports both a raw secret string and a JSON blob containing
    ``OPENAI_API_KEY``.
    """
    import json

    import boto3

    client = boto3.client("secretsmanager", region_name=region)
    resp = client.get_secret_value(SecretId=secret_name)
    raw = resp.get("SecretString", "")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get("OPENAI_API_KEY") or data.get("openai_api_key") or raw
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (one per process)."""
    return Settings()  # type: ignore[call-arg]


def reset_settings_cache() -> None:
    """Clear cached settings (used in tests)."""
    get_settings.cache_clear()
    _fetch_secret.cache_clear()
    os.environ.pop("VECTORRAG_CACHE_BUST", None)
