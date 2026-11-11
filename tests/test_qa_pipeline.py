
import pytest

from app.document_loader import DocumentResult
from app.qa_pipeline import answer_question


class FakeChunk:
    def __init__(self, text, filename, chunk_index, score=0.9):
        self.text = text
        self.score = score
        self.metadata = {
            "filename": filename,
            "chunk_index": chunk_index,
        }


class FakeRetriever:
    def __init__(self):
        self.indexed_text = None
        self.filename = None

    def build_index(self, text, filename="document"):
        self.indexed_text = text
        self.filename = filename
        return 1

    def search(self, query, top_k=3):
        return [
            FakeChunk(
                "Total sales were 500 units.",
                self.filename,
                0,
            )
        ][:top_k]


@pytest.fixture
def sample_document():
    return DocumentResult(
        filename="sales_report.pdf",
        file_type="pdf",
        text="The report contains sales information.",
    )


def test_answer_question_returns_answer_and_sources(
    monkeypatch,
    sample_document,
):
    monkeypatch.setattr(
        "app.qa_pipeline.DocumentRetriever",
        FakeRetriever,
    )

    def fake_generate_response(prompt):
        assert "Total sales were 500 units." in prompt
        assert "What were the sales?" in prompt
        return "Sales were 500 units. [Source 1]"

    monkeypatch.setattr(
        "app.qa_pipeline.generate_response",
        fake_generate_response,
    )

    result = answer_question(
        sample_document,
        "What were the sales?",
    )

    assert result.answer == "Sales were 500 units. [Source 1]"
    assert result.sources[0]["filename"] == "sales_report.pdf"
    assert result.sources[0]["chunk_index"] == 0


def test_rejects_empty_question(sample_document):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(sample_document, "  ")


def test_rejects_empty_document():
    document = DocumentResult(
        filename="empty.pdf",
        file_type="pdf",
        text=" ",
    )

    with pytest.raises(ValueError, match="no searchable text"):
        answer_question(document, "Summarize this document")


def test_rejects_invalid_top_k(sample_document):
    with pytest.raises(ValueError, match="positive integer"):
        answer_question(sample_document, "Question?", top_k=0)

    with pytest.raises(ValueError, match="positive integer"):
        answer_question(sample_document, "Question?", top_k=1.5)


def test_rejects_no_retrieved_chunks(
    monkeypatch,
    sample_document,
):
    class EmptyRetriever(FakeRetriever):
        def search(self, query, top_k=3):
            return []

    monkeypatch.setattr(
        "app.qa_pipeline.DocumentRetriever",
        EmptyRetriever,
    )

    with pytest.raises(ValueError, match="No relevant document"):
        answer_question(sample_document, "Question?")
