"""Security & safety utilities.

Covers the input side of the pipeline:
  * length limits (DoS / cost protection)
  * prompt-injection heuristics (retrieved text and user input are untrusted)
  * PII redaction before anything is logged

These are defence-in-depth heuristics, not a guarantee. The strongest control is
the grounded system prompt (see ``prompts.py``) which instructs the model to treat
retrieved content as data, never as instructions.
"""

from __future__ import annotations

import re

MAX_QUERY_CHARS = 4000

# Patterns that frequently appear in prompt-injection attempts embedded in
# documents or user input. Matches raise the injection flag; they do not block
# outright (the grounded prompt is the real defence) but are logged for review.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(system|previous)\s+prompt", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"\bact\s+as\b", re.I),
    re.compile(r"developer\s+mode", re.I),
]

# PII patterns for redaction in logs.
_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[API_KEY]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[CARD]"),
]


class SecurityError(ValueError):
    """Raised when input fails a hard validation check."""


def sanitize_query(query: str) -> str:
    """Validate and normalise a user query.

    Raises:
        SecurityError: if the query is empty or exceeds the length limit.
    """
    if query is None:
        raise SecurityError("query must not be None")
    query = query.strip()
    if not query:
        raise SecurityError("query must not be empty")
    if len(query) > MAX_QUERY_CHARS:
        raise SecurityError(f"query exceeds {MAX_QUERY_CHARS} characters")
    # Strip control characters that can be used to smuggle instructions.
    return "".join(ch for ch in query if ch == "\n" or ch == "\t" or ord(ch) >= 32)


def detect_injection(text: str) -> bool:
    """Return True if ``text`` matches a known prompt-injection heuristic."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def redact_pii(text: str) -> str:
    """Redact common PII/secrets so text is safe to log."""
    for pattern, repl in _PII_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def neutralize_context(text: str) -> str:
    """Reduce the chance that retrieved text is interpreted as an instruction.

    We do not rewrite the content (that would distort the source), but we collapse
    obvious instruction-injection markers so they cannot hijack the conversation.
    """
    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[redacted-instruction]", cleaned)
    return cleaned
