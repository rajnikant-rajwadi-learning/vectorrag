"""RAG orchestration: the public entry point that ties everything together.

Flow:
  sanitize -> retrieve -> (abstain if no context) -> build grounded prompt ->
  call LLM -> return answer + citations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .clients import build_embedder, build_openai_client, build_vector_store
from .config import Settings, get_settings
from .llm import ChatLLM
from .logging_config import get_logger
from .memory import ConversationMemory
from .prompts import SYSTEM_PROMPT, build_context_block, build_user_prompt
from .retriever import Retriever, Snippet
from .security import detect_injection, redact_pii, sanitize_query

log = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the provided filings to answer that."
)


@dataclass
class Source:
    id: str
    source: str
    page: int | None
    score: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    grounded: bool = True  # False when we abstained for lack of context
    latency_ms: int = 0


class RAGEngine:
    """Reusable engine. Construct once (e.g. per Lambda container) and reuse."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        client = build_openai_client(self._settings)
        self._retriever = Retriever(
            embedder=build_embedder(client, self._settings),
            store=build_vector_store(self._settings),
            settings=self._settings,
        )
        self._llm = ChatLLM(
            client=client,
            model=self._settings.chat_model,
            temperature=self._settings.temperature,
            max_output_tokens=self._settings.max_output_tokens,
        )

    def answer(
        self,
        question: str,
        memory: ConversationMemory | None = None,
    ) -> RAGResponse:
        start = time.perf_counter()
        question = sanitize_query(question)

        if detect_injection(question):
            log.warning("possible_injection_in_query", query=redact_pii(question)[:200])

        snippets = self._retriever.retrieve(question)

        # Anti-hallucination guardrail: no relevant context => abstain, don't ask the
        # model to invent an answer.
        if not snippets:
            log.info("abstain_no_context", query=redact_pii(question)[:200])
            return RAGResponse(
                answer=NO_CONTEXT_ANSWER,
                sources=[],
                grounded=False,
                latency_ms=_ms(start),
            )

        context_block = build_context_block(
            [{"id": s.id, "text": s.text, "metadata": s.metadata} for s in snippets]
        )
        user_prompt = build_user_prompt(question, context_block)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memory is not None:
            messages.extend(memory.as_messages())
        messages.append({"role": "user", "content": user_prompt})

        answer = self._llm.complete(messages)

        if memory is not None:
            memory.add("user", question)
            memory.add("assistant", answer)

        response = RAGResponse(
            answer=answer,
            sources=[_to_source(s) for s in snippets],
            grounded=True,
            latency_ms=_ms(start),
        )
        log.info(
            "answered",
            latency_ms=response.latency_ms,
            sources=len(response.sources),
            top_score=round(snippets[0].score, 3),
        )
        return response


def _to_source(s: Snippet) -> Source:
    return Source(
        id=s.id,
        source=s.metadata.get("source", "unknown"),
        page=s.metadata.get("page"),
        score=round(s.score, 3),
    )


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
