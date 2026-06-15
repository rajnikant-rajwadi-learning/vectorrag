"""Load 10-Q filings from PDF or SEC EDGAR HTML into normalized text pages.

A "page" is the smallest addressable unit we attach to a chunk's metadata so the
model can cite where a fact came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoadedPage:
    """A single page/section of text from a source document."""

    source: str
    page: int
    text: str
    metadata: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    """Normalise whitespace; SEC filings are full of ragged spacing."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[LoadedPage]:
    """Load a PDF 10-Q, one LoadedPage per PDF page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[LoadedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append(LoadedPage(source=path.name, page=i, text=text))
    return pages


def load_html(path: Path) -> list[LoadedPage]:
    """Load an SEC EDGAR HTML 10-Q as a single logical document.

    HTML filings have no real page breaks, so we emit one page (page=1); the
    chunker handles splitting. Tables are flattened to readable text.
    """
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = _clean_text(soup.get_text(separator="\n"))
    return [LoadedPage(source=path.name, page=1, text=text)] if text else []


def load_document(path: str | Path) -> list[LoadedPage]:
    """Dispatch on file extension to load any supported 10-Q file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".html", ".htm"}:
        return load_html(path)
    if suffix in {".txt", ".md"}:
        return [LoadedPage(source=path.name, page=1, text=_clean_text(path.read_text("utf-8")))]
    raise ValueError(f"Unsupported file type: {suffix}")
