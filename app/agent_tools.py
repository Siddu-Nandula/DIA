
"""Tools the Document Intelligence Agent can execute."""

import re
from dataclasses import replace

import pandas as pd

from app.pdf_data_extractor import extract_pdf_dataframe
from app.analytics_qa import answer_analytics_question
from app.chart_generator import (
    generate_bar_chart,
    generate_histogram,
    generate_line_chart,
)
from app.data_analytics import get_dataframe
from app.document_loader import DocumentResult


def _detect_sheet_name(
    document: DocumentResult,
    question: str,
) -> str | None:
    """Find an Excel worksheet name mentioned in the question."""

    if document.file_type.lower() != "xlsx":
        return None

    sheet_names = document.metadata.get("sheet_names", [])
    q = question.lower()

    for sheet_name in sorted(sheet_names, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(sheet_name.lower())}(?!\w)"
        if re.search(pattern, q):
            return sheet_name

    return None


def _detect_row_limit(question: str) -> int | None:
    """Detect requests such as 'first 10 rows'."""

    patterns = [
        r"\bfirst\s+(\d+)\s+rows?\b",
        r"\btop\s+(\d+)\s+rows?\b",
        r"\b(\d+)\s+rows?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question.lower())
        if match:
            return int(match.group(1))

    return None


def _normalize_column_name(name: str) -> str:
    """Normalize a column name for flexible matching."""

    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _match_column(
    requested_text: str,
    columns: list[str],
) -> str | None:
    """Match a requested column name without guessing synonyms."""

    normalized_question = _normalize_column_name(requested_text)

    if not normalized_question:
        return None

    for column in sorted(columns, key=len, reverse=True):
        normalized_column = _normalize_column_name(column)

        if not normalized_column:
            continue

        pattern = (
            rf"(?<![a-z0-9]){re.escape(str(column).lower())}"
            rf"(?![a-z0-9])"
        )

        if re.search(pattern, requested_text.lower()):
            return column

        if normalized_question == normalized_column:
            return column

    return None


def _find_requested_column(
    question: str,
    columns: list[str],
) -> str | None:
    """Find an explicitly mentioned column, including simple plurals."""

    q = question.lower()

    for column in sorted(columns, key=len, reverse=True):
        column_name = re.escape(str(column).lower())

        patterns = (
            rf"(?<![a-z0-9]){column_name}(?![a-z0-9])",
            rf"(?<![a-z0-9]){column_name}s(?![a-z0-9])",
        )

        if any(re.search(pattern, q) for pattern in patterns):
            return column

    return None


def _is_identifier_column(column: str) -> bool:
    """Identify numeric columns that represent record identifiers."""

    normalized = _normalize_column_name(column)

    identifier_names = {
        "id",
        "index",
        "row",
        "rownumber",
        "serial",
        "serialnumber",
        "ordernumber",
        "invoicenumber",
        "transactionnumber",
        "recordnumber",
    }

    return (
        normalized in identifier_names
        or normalized.endswith("id")
    )


def _select_default_metric(
    numeric_columns: list[str],
) -> str | None:
    """Select a meaningful numeric metric, avoiding identifier columns."""

    if not numeric_columns:
        return None

    eligible_columns = [
        column
        for column in numeric_columns
        if not _is_identifier_column(column)
    ]

    if not eligible_columns:
        return None

    preferred_metrics = [
        "revenue",
        "sales",
        "total",
        "amount",
        "profit",
        "income",
        "earnings",
        "price",
        "units",
        "quantity",
        "count",
    ]

    normalized_columns = {
        _normalize_column_name(column): column
        for column in eligible_columns
    }

    for preferred in preferred_metrics:
        if preferred in normalized_columns:
            return normalized_columns[preferred]

    return eligible_columns[0]


def _select_default_category(
    categorical_columns: list[str],
) -> str | None:
    """Select a useful categorical column automatically."""

    if not categorical_columns:
        return None

    preferred_categories = [
        "product",
        "category",
        "region",
        "department",
        "customer",
        "name",
        "type",
        "brand",
    ]

    normalized_columns = {
        _normalize_column_name(column): column
        for column in categorical_columns
    }

    for preferred in preferred_categories:
        if preferred in normalized_columns:
            return normalized_columns[preferred]

    # Avoid choosing a date column as a default bar category.
    non_date_columns = [
        column
        for column in categorical_columns
        if "date" not in _normalize_column_name(column)
        and "time" not in _normalize_column_name(column)
    ]

    if non_date_columns:
        return non_date_columns[0]

    return categorical_columns[0]


def _detect_date_column(
    df: pd.DataFrame,
    question: str,
) -> str | None:
    """Detect a suitable date or time column."""

    date_candidates = []

    # First, identify columns with date/time names or datetime types.
    for column in df.columns:
        normalized = _normalize_column_name(column)

        if (
            pd.api.types.is_datetime64_any_dtype(df[column])
            or any(
                word in normalized
                for word in (
                    "date",
                    "datetime",
                    "timestamp",
                    "time",
                )
            )
        ):
            date_candidates.append(str(column))

    # If no obvious date column exists, inspect string values.
    if not date_candidates:
        for column in df.select_dtypes(
            include=["object", "string"]
        ).columns:
            values = df[column].dropna()

            if values.empty:
                continue

            date_like_ratio = values.astype(str).str.contains(
                r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}",
                regex=True,
            ).mean()

            if date_like_ratio < 0.8:
                continue

            parsed = pd.to_datetime(
                values,
                errors="coerce",
            )

            if parsed.notna().mean() >= 0.8:
                date_candidates.append(str(column))

    if not date_candidates:
        return None

    # Prefer a date column explicitly mentioned in the question.
    requested = _find_requested_column(
        question,
        date_candidates,
    )

    if requested:
        return requested

    # Prefer common date/time column names.
    preferred_names = [
        "date",
        "datetime",
        "timestamp",
        "time",
    ]

    normalized_candidates = {
        _normalize_column_name(column): column
        for column in date_candidates
    }

    for preferred in preferred_names:
        if preferred in normalized_candidates:
            return normalized_candidates[preferred]

    return date_candidates[0]


def _detect_operation(question: str) -> str:
    """Detect the requested aggregation operation."""

    q = question.lower()

    if any(word in q for word in ("average", "mean", "avg")):
        return "mean"

    if any(word in q for word in ("count", "number of", "frequency")):
        return "count"

    return "sum"


def _detect_time_frequency(question: str) -> str | None:
    """Detect the requested time frequency for trend charts."""

    q = question.lower()

    frequency_patterns = {
        "monthly": (
            r"\bmonthly\b",
            r"\bmonth by month\b",
            r"\bby month\b",
        ),
        "weekly": (
            r"\bweekly\b",
            r"\bweek by week\b",
            r"\bby week\b",
        ),
        "daily": (
            r"\bdaily\b",
            r"\bday by day\b",
            r"\bby day\b",
        ),
        "yearly": (
            r"\byearly\b",
            r"\byear by year\b",
            r"\bby year\b",
        ),
    }

    frequencies = {
        "monthly": "MS",
        "weekly": "W",
        "daily": "D",
        "yearly": "YS",
    }

    for name, patterns in frequency_patterns.items():
        if any(re.search(pattern, q) for pattern in patterns):
            return frequencies[name]

    return None


def _is_line_chart_request(question: str) -> bool:
    """Identify requests for trends and time-series charts."""

    q = question.lower()

    line_chart_terms = (
        "line chart",
        "line graph",
        "trend",
        "trends",
        "over time",
        "time series",
        "month by month",
        "week by week",
        "year by year",
        "monthly",
        "daily",
        "weekly",
        "yearly",
        "by date",
        "by month",
        "by day",
        "by year",
    )

    return any(term in q for term in line_chart_terms)



def _is_chart_request(question: str) -> bool:
    """Identify requests that explicitly ask for visualizations."""

    q = question.lower()

    return _is_line_chart_request(question) or any(
        word in q
        for word in (
            "chart",
            "charts",
            "plot",
            "graph",
            "visualize",
            "visualise",
            "histogram",
            "distribution",
            "compare",
            "comparison",
        )
    )




def execute_tool(
    document: DocumentResult,
    question: str,
) -> str:
    """Execute a supported analytics or chart tool."""

    q = question.strip().lower()

    if not q:
        raise ValueError("Question cannot be empty.")

    sheet_name = _detect_sheet_name(document, question)

    if not _is_chart_request(question):
        return answer_analytics_question(
            document,
            question,
            sheet_name=sheet_name,
        )

    # Extract the requested document data.
    if document.file_type.lower() == "pdf":
        df = extract_pdf_dataframe(document, question)
    else:
        df = get_dataframe(
            document,
            sheet_name=sheet_name,
        )

    if df.empty:
        return "Cannot generate charts because the selected data is empty."

    # Apply a requested row limit.
    row_limit = _detect_row_limit(question)

    if row_limit is not None:
        if row_limit <= 0:
            return "The requested row count must be greater than zero."

        df = df.head(row_limit).copy()
    else:
        df = df.copy()

    # Create a temporary document containing only the selected data.
    if document.file_type.lower() == "xlsx" and sheet_name:
        chart_tables = {sheet_name: df}
    else:
        chart_tables = [df]

    chart_document = replace(
        document,
        tables=chart_tables,
    )

    numeric_columns = [
        str(col)
        for col in df.select_dtypes(include="number").columns
    ]

    categorical_columns = [
        str(col)
        for col in df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    ]

    if not numeric_columns:
        return (
            "I couldn't identify a numeric metric for the requested chart. "
            "The selected worksheet contains no numeric columns."
        )

    histogram_requested = any(
        word in q
        for word in ("histogram", "distribution")
    )

    # Identify an explicitly requested numeric metric.
    metric = _find_requested_column(
        question,
        numeric_columns,
    )

    # For histograms, check whether the user explicitly requested
    # a non-numeric column instead of silently selecting another metric.
    if histogram_requested and metric is None:
        requested_column = _find_requested_column(
            question,
            [str(col) for col in df.columns],
        )

        if requested_column is not None:
            return (
                f"'{requested_column}' is not a numeric column, "
                "so a histogram cannot be generated from it. "
                f"Available numeric columns: {', '.join(numeric_columns)}."
            )

    # Automatically select a meaningful metric only when none was requested.
    if metric is None:
        metric = _select_default_metric(numeric_columns)

        if metric is None:
            return (
                "I found numeric columns, but they appear to be "
                "identifiers rather than useful chart metrics. "
                f"Available numeric columns: {', '.join(numeric_columns)}."
            )

    # Identify an explicitly requested categorical column.
    category = _find_requested_column(
        question,
        categorical_columns,
    )

    explicit_relationship = any(
        phrase in q
        for phrase in (
            " by ",
            " against ",
            " versus ",
            " vs ",
            " across ",
        )
    )

    operation = _detect_operation(question)

    # ---------------------------------------------------------
    # LINE CHART MODE
    # ---------------------------------------------------------
    if _is_line_chart_request(question):

        date_column = _detect_date_column(
            df,
            question,
        )

        if date_column is None:
            return (
                "I couldn't identify a date or time column "
                "for the requested trend chart."
            )

        df[date_column] = pd.to_datetime(
            df[date_column],
            errors="coerce",
        )

        df = df.dropna(
            subset=[date_column]
        ).copy()

        if df.empty:
            return (
                "No valid date values were found "
                "for the requested line chart."
            )

        # Rebuild the temporary document using cleaned data.
        if document.file_type.lower() == "xlsx" and sheet_name:
            chart_tables = {sheet_name: df}
        else:
            chart_tables = [df]

        chart_document = replace(
            document,
            tables=chart_tables,
        )

        frequency = _detect_time_frequency(question)

        path = generate_line_chart(
            chart_document,
            x_axis=date_column,
            metric=metric,
            filename="agent_line_chart.png",
            sheet_name=sheet_name,
            frequency=frequency,
            operation=operation,
        )

        return (
            "Line chart generated successfully.\n"
            f"Date column: {date_column}\n"
            f"Metric: {metric}\n"
            f"Aggregation: {operation}\n"
            f"Time frequency: {frequency or 'Original data'}\n"
            f"Rows used: {len(df)}\n"
            f"Worksheet: {sheet_name or 'default table'}\n"
            f"File: {path}"
        )

    # ---------------------------------------------------------
    # HISTOGRAM MODE
    # ---------------------------------------------------------
    if histogram_requested:
        path = generate_histogram(
            chart_document,
            metric=metric,
            filename="agent_histogram.png",
            sheet_name=sheet_name,
        )

        return (
            "Histogram generated successfully.\n"
            f"Metric: {metric}\n"
            f"Rows used: {len(df)}\n"
            f"Worksheet: {sheet_name or 'default table'}\n"
            f"File: {path}"
        )

    # ---------------------------------------------------------
    # AUTOMATIC CATEGORY SELECTION
    # ---------------------------------------------------------
    if category is None:
        if explicit_relationship:
            available = ", ".join(categorical_columns) or "none"

            return (
                "I couldn't identify the requested category column. "
                f"Available category columns: {available}."
            )

        category = _select_default_category(
            categorical_columns
        )

    # ---------------------------------------------------------
    # HISTOGRAM FALLBACK
    # ---------------------------------------------------------
    if category is None:
        path = generate_histogram(
            chart_document,
            metric=metric,
            filename="agent_histogram.png",
            sheet_name=sheet_name,
        )

        return (
            "Histogram generated successfully.\n"
            f"Metric: {metric}\n"
            f"Rows used: {len(df)}\n"
            f"Worksheet: {sheet_name or 'default table'}\n"
            f"File: {path}"
        )

    # ---------------------------------------------------------
    # BAR CHART MODE
    # ---------------------------------------------------------
    path = generate_bar_chart(
        chart_document,
        category=category,
        metric=metric,
        operation=operation,
        filename="agent_chart.png",
        sheet_name=sheet_name,
    )

    return (
        "Chart generated successfully.\n"
        f"Category: {category}\n"
        f"Metric: {metric}\n"
        f"Operation: {operation}\n"
        f"Rows used: {len(df)}\n"
        f"Worksheet: {sheet_name or 'default table'}\n"
        f"File: {path}"
    )
