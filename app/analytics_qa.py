
"""Answer numerical questions using exact pandas calculations."""

import pandas as pd
import re

from app.data_analytics import (
    analyze_dataframe,
    get_dataframe,
    group_by_summary,
)
from app.document_loader import DocumentResult
from app.pdf_data_extractor import extract_pdf_dataframe


CATEGORY_COLUMNS = (
    "region",
    "product",
    "category",
    "rep",
    "status",
)


def _requested_filter_column(question: str) -> str | None:
    """Identify a category column explicitly mentioned in a filter question."""

    q = question.casefold()

    has_filter_word = bool(
        re.search(r"\b(in|for|from|where|within)\b", q)
    )

    if not has_filter_word:
        return None

    for column in CATEGORY_COLUMNS:
        if re.search(rf"\b{re.escape(column)}\b", q):
            return column

    return None


def _answer_summary_table(df, question: str) -> str | None:
    """Answer questions from PDF Metric/Value summary tables."""

    column_lookup = {
        str(column).strip().casefold(): column
        for column in df.columns
    }

    if not {"metric", "value"}.issubset(column_lookup):
        return None

    metric_column = column_lookup["metric"]
    value_column = column_lookup["value"]

    q = question.casefold()

    metric_aliases = {
        "revenue": ("revenue", "sales"),
        "profit": ("profit",),
        "units": ("units",),
        "orders": ("orders",),
    }

    selected_metric = None

    for metric, aliases in metric_aliases.items():
        if any(
            re.search(rf"\b{re.escape(alias)}\b", q)
            for alias in aliases
        ):
            selected_metric = metric
            break

    if selected_metric is None:
        return None

    wants_average = bool(
        re.search(r"\b(average|mean)\b", q)
    )

    metric_values = df[metric_column].astype("string")

    if wants_average:
        patterns = {
            "revenue": r"\baverage\s+(?:order\s+)?revenue\b",
            "profit": r"\baverage\s+profit\b",
            "units": r"\baverage\s+units\b",
            "orders": r"\baverage\s+orders\b",
        }
    else:
        patterns = {
            "revenue": r"\btotal\s+revenue\b|\boverall\s+revenue\b",
            "profit": r"\btotal\s+profit\b|\boverall\s+profit\b",
            "units": r"\btotal\s+units\b|\boverall\s+units\b",
            "orders": r"\btotal\s+orders\b|\boverall\s+orders\b",
        }

    matches = df[
        metric_values.str.contains(
            patterns[selected_metric],
            case=False,
            regex=True,
            na=False,
        )
    ]

    if matches.empty:
        return None

    value = pd.to_numeric(
        matches.iloc[0][value_column],
        errors="coerce",
    )

    if pd.isna(value):
        return None

    label = "Average" if wants_average else "Total"

    return f"{label} {selected_metric}: {value:,.2f}"

def _requested_filter_columns(question: str) -> list[str]:
    """Find all category columns explicitly requested in a question."""

    q = question.casefold()

    has_filter_word = bool(
        re.search(r"\b(in|for|from|where|within)\b", q)
    )

    if not has_filter_word:
        return []

    return [
        column
        for column in CATEGORY_COLUMNS
        if re.search(rf"\b{re.escape(column)}\b", q)
    ]


def _apply_category_filter(
    df: pd.DataFrame,
    question: str,
) -> tuple[pd.DataFrame, str | None]:
    """Apply one or more category filters mentioned in a question."""

    q = question.casefold()

    categorical_columns = [
        str(column)
        for column in df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    ]

    requested_columns = _requested_filter_columns(q)

    if not requested_columns:
        return df, None

    filtered_df = df.copy()

    for requested_column in requested_columns:

        actual_column = next(
            (
                column
                for column in categorical_columns
                if column.casefold() == requested_column
            ),
            None,
        )

        if actual_column is None:
            return df, (
                f"The selected table does not contain the "
                f"'{requested_column}' column. Detailed table data "
                "is required to answer this filtered question."
            )

        values = (
            df[actual_column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        matching_values = [
            value
            for value in values
            if re.search(
                rf"\b{re.escape(value.casefold())}\b",
                q,
            )
        ]

        if len(matching_values) > 1:
            return df, (
                f"Multiple values were found for {actual_column}. "
                "Please specify one value."
            )

        if len(matching_values) == 0:
            return df, (
                f"No matching value was found for {actual_column} "
                "in your question. Please check the category name."
            )

        filter_value = matching_values[0]

        filtered_df = filtered_df[
            filtered_df[actual_column]
            .astype(str)
            .str.strip()
            .str.casefold()
            == filter_value.casefold()
        ].copy()

        if filtered_df.empty:
            return filtered_df, (
                f"No records found for "
                f"{actual_column} = {filter_value}."
            )

    return filtered_df, None

def answer_analytics_question(
    document: DocumentResult,
    question: str,
    sheet_name: str | None = None,
) -> str:
    """Answer numerical questions about PDF, CSV, or Excel data."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    q = question.strip().lower()

    if document.file_type.lower() == "pdf":
        df = extract_pdf_dataframe(document, question)
    else:
        df = get_dataframe(document, sheet_name=sheet_name)

    print("\n--- ANALYTICS DATAFRAME ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Apply category filtering BEFORE summary-table calculations.
    df, filter_error = _apply_category_filter(df, q)

    if filter_error:
        return filter_error

    # Handle PDF summary tables only when no category filter is required.
    summary_answer = _answer_summary_table(df, q)

    if summary_answer is not None:
        return summary_answer

    # Count rows and columns.
    if re.search(r"\b(rows?|records|entries)\b", q) and re.search(
        r"\b(how many|number|count|total)\b", q
    ):
        return f"The document contains {len(df)} rows."

    if re.search(r"\b(columns?|fields)\b", q) and re.search(
        r"\b(how many|number|count|total)\b", q
    ):
        return f"The document contains {len(df.columns)} columns."

    # Find numeric columns mentioned in the question.
    numeric_columns = [
        str(column)
        for column in df.select_dtypes(include="number").columns
    ]

    metric = next(
        (
            column
            for column in numeric_columns
            if re.search(rf"\b{re.escape(column.lower())}\b", q)
        ),
        None,
    )

    # Handle generic revenue or sales questions.
    if metric is None:
        for column in numeric_columns:
            if column.lower() in {"revenue", "sales"}:
                if re.search(r"\b(revenue|sales)\b", q):
                    metric = column
                    break

    if metric is None:
        return (
            "I couldn't identify a supported numerical metric in your "
            "question. Try asking about rows, columns, revenue, or units."
        )

    # Grouped analysis.
    categorical_columns = [
        str(column)
        for column in df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    ]

    group_by = next(
        (
            column
            for column in categorical_columns
            if re.search(rf"\b{re.escape(column.lower())}\b", q)
        ),
        None,
    )

    ranking_words = {
        "highest", "maximum", "max",
        "lowest", "minimum", "min",
        "most", "least", "top", "bottom",
    }

    if group_by and any(word in q.split() for word in ranking_words):
        operation = (
            "mean"
            if re.search(r"\b(average|mean)\b", q)
            else "sum"
        )

        summary = group_by_summary(
            df,
            group_by=group_by,
            metric=metric,
            operation=operation,
        )

        if summary.empty:
            return "There is no data available for this calculation."

        if any(
            word in q.split()
            for word in ("lowest", "minimum", "min", "least", "bottom")
        ):
            best = summary.loc[summary[metric].idxmin()]
            ranking = "lowest"
        else:
            best = summary.loc[summary[metric].idxmax()]
            ranking = "highest"

        label = "average" if operation == "mean" else "total"

        return (
            f"{group_by.title()} with {ranking} {label} "
            f"{metric}: {best[group_by]} "
            f"({best[metric]:,.2f})"
        )

    if group_by and re.search(r"\b(by|per|each|across|among)\b", q):
        operation = (
            "mean"
            if re.search(r"\b(average|mean)\b", q)
            else "sum"
        )

        summary = group_by_summary(
            df,
            group_by=group_by,
            metric=metric,
            operation=operation,
        )

        if summary.empty:
            return "There is no data available for this calculation."

        rows = [
            f"{row[group_by]}: {row[metric]:,.2f}"
            for _, row in summary.iterrows()
        ]

        label = "average" if operation == "mean" else "total"

        return (
            f"{label.title()} {metric} by {group_by}:\n"
            + "\n".join(rows)
        )

    # Calculate statistics from the selected DataFrame.
    stats = analyze_dataframe(df)["numeric_summary"].get(metric)

    if stats is None:
        return f"No numeric values are available for '{metric}'."

    if re.search(r"\b(average|mean)\b", q):
        return f"Average {metric}: {stats['mean']:,.2f}"

    if re.search(r"\b(highest|maximum|max)\b", q):
        return f"Highest {metric}: {stats['max']:,.2f}"

    if re.search(r"\b(lowest|minimum|min)\b", q):
        return f"Lowest {metric}: {stats['min']:,.2f}"

    if re.search(r"\b(how many|count)\b", q) and metric == "units":
        return f"Total units: {stats['sum']:,.0f}"

    if re.search(r"\b(total|sum|overall)\b", q):
        return f"Total {metric}: {stats['sum']:,.2f}"

    return (
        f"For {metric}: total = {stats['sum']:,.2f}, "
        f"average = {stats['mean']:,.2f}, "
        f"minimum = {stats['min']:,.2f}, "
        f"maximum = {stats['max']:,.2f}."
    )
