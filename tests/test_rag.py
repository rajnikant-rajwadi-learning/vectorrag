"""RAG engine tests with fully mocked OpenAI + vector store (no network)."""

from __future__ import annotations

import pytest

from vectorrag.config import Settings
from vectorrag.rag import NO_CONTEXT_ANSWER, RAGEngine
from vectorrag.retriever import Snippet
from vectorrag.security import SecurityError


class _FakeRetriever:
    def __init__(self, snippets):
        self._snippets = snippets

    def retrieve(self, query):  # noqa: ARG002
        return self._snippets


class _FakeLLM:
    def __init__(self, reply="Revenue was $10M [S1]."):
        self.reply = reply
        self.last_messages = None

    def complete(self, messages):
        self.last_messages = messages
        return self.reply


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return Settings(_env_file=None)  # type: ignore[call-arg]


def _engine_with(monkeypatch, settings, snippets, reply="ans [S1]"):
    # Avoid constructing real OpenAI/Chroma clients.
    monkeypatch.setattr(RAGEngine, "__init__", lambda self, s=None: None)
    engine = RAGEngine()
    engine._settings = settings
    engine._retriever = _FakeRetriever(snippets)
    engine._llm = _FakeLLM(reply)
    return engine


def test_abstains_without_context(monkeypatch, settings):
    engine = _engine_with(monkeypatch, settings, snippets=[])
    resp = engine.answer("What was revenue?")
    assert resp.answer == NO_CONTEXT_ANSWER
    assert resp.grounded is False
    assert resp.sources == []


def test_answers_with_context(monkeypatch, settings):
    snippets = [
        Snippet(id="S1", text="Revenue was $10M.", metadata={"source": "acme.pdf", "page": 3},
                score=0.9)
    ]
    engine = _engine_with(monkeypatch, settings, snippets, reply="Revenue was $10M [S1].")
    resp = engine.answer("What was revenue?")
    assert resp.grounded is True
    assert "[S1]" in resp.answer
    assert resp.sources[0].source == "acme.pdf"
    assert resp.sources[0].page == 3


def test_system_prompt_present(monkeypatch, settings):
    snippets = [Snippet(id="S1", text="x", metadata={"source": "a", "page": 1}, score=0.8)]
    engine = _engine_with(monkeypatch, settings, snippets)
    engine.answer("q?")
    msgs = engine._llm.last_messages
    assert msgs[0]["role"] == "system"
    assert "ONLY" in msgs[0]["content"]


def test_rejects_empty_query(monkeypatch, settings):
    engine = _engine_with(monkeypatch, settings, snippets=[])
    with pytest.raises(SecurityError):
        engine.answer("   ")
