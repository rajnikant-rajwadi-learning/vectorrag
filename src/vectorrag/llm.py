"""OpenAI chat completion wrapper with retry and deterministic defaults.

Temperature defaults to 0.0 (set in config) to minimise hallucination and make
answers reproducible for the same context.
"""

from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .logging_config import get_logger

log = get_logger(__name__)


class ChatLLM:
    def __init__(self, client: OpenAI, model: str, temperature: float, max_output_tokens: int):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def complete(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_output_tokens,
        )
        usage = resp.usage
        log.info(
            "chat_completion",
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
        return (resp.choices[0].message.content or "").strip()
