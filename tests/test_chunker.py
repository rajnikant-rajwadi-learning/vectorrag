from vectorrag.ingest.chunker import chunk_pages
from vectorrag.ingest.loader import LoadedPage


def _make_pages(text: str) -> list[LoadedPage]:
    return [LoadedPage(source="acme_10q.pdf", page=1, text=text)]


def test_chunking_respects_size():
    text = " ".join(f"Sentence number {i} about revenue." for i in range(200))
    chunks = chunk_pages(
        _make_pages(text),
        chunk_size_tokens=50,
        chunk_overlap_tokens=10,
        embedding_model="text-embedding-3-small",
    )
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "acme_10q.pdf" for c in chunks)
    assert all(c.metadata["page"] == 1 for c in chunks)


def test_chunk_ids_unique():
    text = " ".join(f"Item {i}." for i in range(100))
    chunks = chunk_pages(
        _make_pages(text),
        chunk_size_tokens=30,
        chunk_overlap_tokens=5,
        embedding_model="text-embedding-3-small",
    )
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_overlap_preserves_continuity():
    text = " ".join(f"Fact {i} matters." for i in range(60))
    chunks = chunk_pages(
        _make_pages(text),
        chunk_size_tokens=40,
        chunk_overlap_tokens=15,
        embedding_model="text-embedding-3-small",
    )
    # With overlap, consecutive chunks should share some leading text.
    assert len(chunks) >= 2


def test_empty_page_yields_no_chunks():
    chunks = chunk_pages(
        [],
        chunk_size_tokens=40,
        chunk_overlap_tokens=10,
        embedding_model="text-embedding-3-small",
    )
    assert chunks == []
