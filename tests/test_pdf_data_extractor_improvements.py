
import pandas as pd
import pytest

from app.document_loader import DocumentResult
from app.pdf_data_extractor import (
    extract_pdf_dataframe,
    _clean_dataframe,
    _extract_json,
)


@pytest.fixture
def pdf_document():
    return DocumentResult(
        filename="marks.pdf",
        file_type="pdf",
        text=(
            "Student Marks\n"
            "Mathematics 85\n"
            "Physics 90\n"
            "Chemistry 78\n"
        ),
        tables=[],
    )


def test_extracts_relevant_existing_pdf_table():
    document = DocumentResult(
        filename="marks.pdf",
        file_type="pdf",
        text="Subject Marks\nMathematics 85\nPhysics 90",
        tables=[
            pd.DataFrame({
                "Subject": ["Mathematics", "Physics"],
                "Marks": ["85", "90"],
            })
        ],
    )

    result = extract_pdf_dataframe(
        document,
        "Create a chart of marks by subject",
    )

    assert list(result.columns) == ["Subject", "Marks"]
    assert result["Marks"].tolist() == [85, 90]


def test_extracts_structured_data_from_pdf_text(
    pdf_document,
    monkeypatch,
):
    def fake_ask_llm(prompt, instructions=None):
        return """
        {
            "columns": ["Subject", "Marks"],
            "rows": [
                ["Mathematics", 85],
                ["Physics", 90],
                ["Chemistry", 78]
            ],
            "evidence": [
                "Mathematics 85",
                "Physics 90",
                "Chemistry 78"
            ]
        }
        """

    monkeypatch.setattr(
        "app.pdf_data_extractor.ask_llm",
        fake_ask_llm,
    )

    result = extract_pdf_dataframe(
        pdf_document,
        "Create a chart of marks by subject",
    )

    assert result["Subject"].tolist() == [
        "Mathematics",
        "Physics",
        "Chemistry",
    ]

    assert result["Marks"].tolist() == [85, 90, 78]


def test_rejects_unverified_extracted_values(
    pdf_document,
    monkeypatch,
):
    def fake_ask_llm(prompt, instructions=None):
        return """
        {
            "columns": ["Subject", "Marks"],
            "rows": [
                ["Mathematics", 100]
            ],
            "evidence": [
                "Mathematics 100"
            ]
        }
        """

    monkeypatch.setattr(
        "app.pdf_data_extractor.ask_llm",
        fake_ask_llm,
    )

    with pytest.raises(
        ValueError,
        match="could not be verified",
    ):
        extract_pdf_dataframe(
            pdf_document,
            "Create a chart of marks by subject",
        )


def test_rejects_pdf_without_requested_data(
    pdf_document,
    monkeypatch,
):
    def fake_ask_llm(prompt, instructions=None):
        return """
        {
            "columns": [],
            "rows": [],
            "evidence": []
        }
        """

    monkeypatch.setattr(
        "app.pdf_data_extractor.ask_llm",
        fake_ask_llm,
    )

    with pytest.raises(
        ValueError,
        match="No suitable chartable data",
    ):
        extract_pdf_dataframe(
            pdf_document,
            "Create a chart of annual revenue",
        )


def test_cleans_numeric_values():
    dataframe = pd.DataFrame({
        "Subject": ["Mathematics", "Physics"],
        "Marks": ["1,200", "900"],
    })

    result = _clean_dataframe(dataframe)

    assert result["Marks"].tolist() == [1200, 900]
    assert pd.api.types.is_numeric_dtype(result["Marks"])


def test_extract_json_from_markdown():
    response = """
    ```json
    {
        "columns": ["Subject", "Marks"],
        "rows": [["Mathematics", 85]],
        "evidence": ["Mathematics 85"]
    }
    ```
    """

    result = _extract_json(response)

    assert result["columns"] == ["Subject", "Marks"]
    assert result["rows"] == [["Mathematics", 85]]


def test_rejects_empty_question(pdf_document):
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        extract_pdf_dataframe(pdf_document, "   ")


def test_rejects_non_pdf_document():
    document = DocumentResult(
        filename="sales.csv",
        file_type="csv",
        text="Sales data",
    )

    with pytest.raises(
        ValueError,
        match="PDF documents only",
    ):
        extract_pdf_dataframe(
            document,
            "Create a chart",
        )
