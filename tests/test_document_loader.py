
import pandas as pd
import pytest
from openpyxl import Workbook
from pypdf import PdfWriter

from app.document_loader import (
    load_csv,
    load_document,
    load_excel,
    load_pdf,
    validate_file,
)


def create_excel_workbook(path):
    """Create a test workbook with two worksheets."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "product": ["Laptop", "Phone"],
                "revenue": [50000, 25000],
            }
        ).to_excel(writer, sheet_name="Sales", index=False)

        pd.DataFrame(
            {
                "region": ["South", "North"],
                "manager": ["Ravi", "Priya"],
            }
        ).to_excel(writer, sheet_name="Regions", index=False)


def test_load_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "20")

    path = tmp_path / "sales.csv"
    path.write_text(
        "product,revenue\nLaptop,50000\nPhone,25000\n",
        encoding="utf-8",
    )

    result = load_csv(path)

    assert result.file_type == "csv"
    assert result.metadata["rows"] == 2
    assert result.metadata["columns"] == 2
    assert isinstance(result.tables[0], pd.DataFrame)
    assert result.tables[0]["revenue"].sum() == 75000


def test_load_document_selects_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\nA,10\nB,20\n", encoding="utf-8")

    result = load_document(path)

    assert result.file_type == "csv"
    assert result.filename == "data.csv"


def test_load_excel_reads_multiple_worksheets(tmp_path):
    path = tmp_path / "sales.xlsx"
    create_excel_workbook(path)

    result = load_excel(path)

    assert result.file_type == "xlsx"
    assert result.filename == "sales.xlsx"
    assert result.metadata["sheet_names"] == ["Sales", "Regions"]
    assert result.metadata["worksheet_count"] == 2
    assert result.metadata["total_rows"] == 4
    assert len(result.tables) == 2
    assert result.tables[0]["revenue"].sum() == 75000
    assert result.tables[1]["manager"].tolist() == ["Ravi", "Priya"]


def test_load_document_selects_excel(tmp_path):
    path = tmp_path / "data.xlsx"
    create_excel_workbook(path)

    result = load_document(path)

    assert result.file_type == "xlsx"
    assert result.metadata["sheet_names"] == ["Sales", "Regions"]


def test_validate_file_accepts_xlsx(tmp_path):
    path = tmp_path / "data.xlsx"
    create_excel_workbook(path)

    assert validate_file(path) == path


def test_load_excel_rejects_csv(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,value\nA,10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires an XLSX"):
        load_excel(path)


def test_invalid_excel_file_raises_error(tmp_path):
    path = tmp_path / "invalid.xlsx"
    path.write_text("This is not a valid Excel workbook.", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read Excel file"):
        load_excel(path)


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_file(tmp_path / "missing.csv")


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_file(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        validate_file(path)


def test_load_csv_rejects_pdf(tmp_path):
    path = tmp_path / "report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    with path.open("wb") as file:
        writer.write(file)

    with pytest.raises(ValueError, match="requires a CSV"):
        load_csv(path)


def test_blank_pdf_is_rejected(tmp_path):
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    with path.open("wb") as file:
        writer.write(file)

    with pytest.raises(ValueError, match="No readable text"):
        load_pdf(path)
