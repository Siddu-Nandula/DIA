from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pdfplumber

from app.config import get_settings


@dataclass
class DocumentResult:
    """Standard result returned by the document loader."""

    filename: str
    file_type: str
    text: str = ""
    tables: list[pd.DataFrame] = field(default_factory=list)
    page_texts: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def validate_file(file_path: str | Path) -> Path:
    """Validate file existence, extension, and size."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError("The supplied path must point to a file.")

    suffix = path.suffix.lower()

    if suffix not in {".pdf", ".csv", ".xlsx"}:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            "Use PDF, CSV, or XLSX."
    )

    if path.stat().st_size == 0:
        raise ValueError("The supplied file is empty.")

    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    if path.stat().st_size > max_bytes:
        raise ValueError(
            f"File exceeds the {settings.max_upload_mb} MB limit."
        )

    return path


def load_csv(file_path: str | Path) -> DocumentResult:
    """Load a CSV file and return its data and a readable summary."""

    path = validate_file(file_path)

    if path.suffix.lower() != ".csv":
        raise ValueError("load_csv() requires a CSV file.")

    try:
        dataframe = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not parse CSV file: {exc}") from exc

    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError("CSV has no data rows or columns.")

    if not dataframe.columns.is_unique:
        raise ValueError("CSV contains duplicate column names.")

    summary = (
        f"Filename: {path.name}\n"
        f"Rows: {len(dataframe)}\n"
        f"Columns: {len(dataframe.columns)}\n"
        f"Column names: {', '.join(map(str, dataframe.columns))}\n\n"
        f"Preview:\n{dataframe.head(10).to_string(index=False)}"
    )

    return DocumentResult(
        filename=path.name,
        file_type="csv",
        text=summary,
        tables=[dataframe],
        metadata={
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "column_names": list(dataframe.columns),
        },
    )

def load_excel(file_path: str | Path) -> DocumentResult:
    """Load an Excel workbook and extract its worksheets."""

    path = validate_file(file_path)

    if path.suffix.lower() != ".xlsx":
        raise ValueError("load_excel() requires an XLSX file.")

    try:
        # Read every worksheet into a dictionary of DataFrames.
        worksheets = pd.read_excel(
            path,
            sheet_name=None,
            engine="openpyxl",
        )
    except Exception as exc:
        raise ValueError(
            f"Could not read Excel file: {exc}"
        ) from exc

    tables = []
    sheet_names = []
    sheet_details = {}
    text_sections = []

    for sheet_name, dataframe in worksheets.items():
        # Skip worksheets with no usable data.
        if dataframe.empty or len(dataframe.columns) == 0:
            continue

        if not dataframe.columns.is_unique:
            raise ValueError(
                f"Worksheet '{sheet_name}' contains duplicate "
                "column names."
            )

        tables.append(dataframe)
        sheet_names.append(sheet_name)

        sheet_details[sheet_name] = {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "column_names": list(dataframe.columns),
        }

        text_sections.append(
            f"Worksheet: {sheet_name}\n"
            f"Rows: {len(dataframe)}\n"
            f"Columns: {len(dataframe.columns)}\n"
            f"Column names: {', '.join(map(str, dataframe.columns))}\n\n"
            f"Preview:\n{dataframe.head(10).to_string(index=False)}"
        )

    if not tables:
        raise ValueError(
            "The Excel workbook contains no worksheets with data."
        )

    return DocumentResult(
        filename=path.name,
        file_type="xlsx",
        text="\n\n".join(text_sections),
        tables=tables,
        metadata={
            "sheet_names": sheet_names,
            "sheet_details": sheet_details,
            "total_rows": sum(
                details["rows"]
                for details in sheet_details.values()
            ),
            "worksheet_count": len(sheet_names),
        },
    )


def load_pdf(file_path: str | Path) -> DocumentResult:
    """Extract page text and merge matching tables across PDF pages."""

    path = validate_file(file_path)

    if path.suffix.lower() != ".pdf":
        raise ValueError("load_pdf() requires a PDF file.")

    page_texts = []
    extracted_tables = []

    try:
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                raise ValueError("PDF contains no pages.")

            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""

                page_texts.append({
                    "page": page_number,
                    "text": page_text,
                })

                for table in page.extract_tables():
                    if not table or len(table) < 2:
                        continue

                    headers = [
                        str(header).strip() if header is not None
                        else f"column_{index + 1}"
                        for index, header in enumerate(table[0])
                    ]

                    rows = table[1:]

                    cleaned_rows = [
                        [
                            value.strip() if isinstance(value, str)
                            else value
                            for value in row
                        ]
                        for row in rows
                        if row and any(
                            value is not None and str(value).strip()
                            for value in row
                        )
                    ]

                    if cleaned_rows:
                        dataframe = pd.DataFrame(
                            cleaned_rows,
                            columns=headers,
                        )

                        extracted_tables.append(dataframe)

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Could not read PDF file: {exc}"
        ) from exc

    # Merge tables having identical column headers.
    merged_tables = {}

    for dataframe in extracted_tables:
        normalized_headers = tuple(
            str(column).strip().casefold()
            for column in dataframe.columns
        )

        if normalized_headers not in merged_tables:
            merged_tables[normalized_headers] = dataframe.copy()
        else:
            merged_tables[normalized_headers] = pd.concat(
                [
                    merged_tables[normalized_headers],
                    dataframe,
                ],
                ignore_index=True,
            )

    extracted_tables = list(merged_tables.values())

    combined_text = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in page_texts
        if item["text"].strip()
    )

    if not combined_text.strip() and not extracted_tables:
        raise ValueError(
            "No readable text or tables found. "
            "The PDF may be scanned or image-only."
        )

    return DocumentResult(
        filename=path.name,
        file_type="pdf",
        text=combined_text,
        tables=extracted_tables,
        page_texts=page_texts,
        metadata={
            "pages": len(page_texts),
            "tables_found": len(extracted_tables),
            "raw_tables_found": len(merged_tables),
            "total_table_rows": sum(
                len(table) for table in extracted_tables
            ),
        },
    )


def load_document(file_path: str | Path) -> DocumentResult:
    """Choose the correct loader based on file extension."""

    path = validate_file(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_csv(path)

    if suffix == ".xlsx":
        return load_excel(path)

    if suffix == ".pdf":
        return load_pdf(path)

    # Defensive check in case supported extensions change.
    raise ValueError(f"Unsupported file type: {suffix}")