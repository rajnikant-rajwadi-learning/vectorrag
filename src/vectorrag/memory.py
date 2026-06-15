"""Token-bounded conversation memory.

Keeps a rolling window of the most recent turns that fit within
``max_history_tokens``. This caps cost and prevents the prompt from growing without
bound across a long chat, while preserving recent context for follow-up questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tokens import count_tokens


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ConversationMemory:
    max_history_tokens: int
    model: str = "gpt-4o-mini"
    _turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        self._trim()

    def _trim(self) -> None:
        """Drop oldest turns until total tokens fit the budget."""
        while self._turns and self._total_tokens() > self.max_history_tokens:
            self._turns.pop(0)

    def _total_tokens(self) -> int:
        return sum(count_tokens(t.content, self.model) for t in self._turns)

    def as_messages(self) -> list[dict]:
        """Return history as OpenAI chat messages (excludes the system prompt)."""
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()
