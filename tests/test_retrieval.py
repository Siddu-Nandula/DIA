
import numpy as np
import pytest

from app.retrieval import DocumentRetriever


def test_split_text_returns_overlapping_chunks():
    retriever = DocumentRetriever(
        chunk_size=10,
        chunk_overlap=2,
    )

    chunks = retriever.split_text("abcdefghijklmnop")

    assert chunks == [
        "abcdefghij",
        "ijklmnop",
    ]


def test_split_text_empty_input():
    retriever = DocumentRetriever()
    assert retriever.split_text("   ") == []


def test_build_index_and_search(monkeypatch):
    retriever = DocumentRetriever()

    def fake_embed(texts):
        vectors = []
        for text in texts:
            if "sales" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    monkeypatch.setattr(retriever, "_embed", fake_embed)

    count = retriever.build_index(
        "Sales increased this year. " * 5,
        filename="report.txt",
    )

    assert count > 0

    results = retriever.search("sales", top_k=1)

    assert len(results) == 1
    assert "Sales" in results[0].text
    assert results[0].metadata["filename"] == "report.txt"


def test_search_before_building_index():
    retriever = DocumentRetriever()

    with pytest.raises(RuntimeError, match="Build the index"):
        retriever.search("What are the sales?")


def test_build_index_rejects_empty_text():
    retriever = DocumentRetriever()

    with pytest.raises(ValueError, match="empty document"):
        retriever.build_index("   ")


def test_search_rejects_empty_query(monkeypatch):
    retriever = DocumentRetriever()
    monkeypatch.setattr(
        retriever,
        "_embed",
        lambda texts: np.ones((len(texts), 3), dtype=np.float32),
    )
    retriever.build_index("Some document content.")

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.search("  ")


def test_invalid_chunk_configuration():
    with pytest.raises(ValueError, match="chunk_size"):
        DocumentRetriever(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap"):
        DocumentRetriever(chunk_size=10, chunk_overlap=10)