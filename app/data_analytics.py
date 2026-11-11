import pandas as pd

from app.document_loader import DocumentResult


def get_dataframe(
    document: DocumentResult,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    """Extract a DataFrame, optionally selecting an Excel worksheet."""

    tables = getattr(document, "tables", None)

    # Support a DataFrame directly, for compatibility.
    if isinstance(tables, pd.DataFrame):
        if sheet_name is not None:
            raise ValueError(
                "Worksheet selection is only supported for Excel workbooks."
            )
        return tables.copy()

    # Support dictionaries of DataFrames, if provided.
    if isinstance(tables, dict):
        if sheet_name is not None:
            if sheet_name not in tables:
                raise ValueError(
                    f"Worksheet '{sheet_name}' does not exist."
                )
            selected = tables[sheet_name]
            if not isinstance(selected, pd.DataFrame):
                raise ValueError(
                    f"Worksheet '{sheet_name}' contains no DataFrame."
                )
            return selected.copy()

        tables = list(tables.values())

    if isinstance(tables, (list, tuple)):
        # Excel loader stores worksheet names and DataFrames in matching order.
        if document.file_type == "xlsx":
            sheet_names = document.metadata.get("sheet_names", [])

            if sheet_name is not None:
                if sheet_name not in sheet_names:
                    available = ", ".join(sheet_names) or "none"
                    raise ValueError(
                        f"Worksheet '{sheet_name}' does not exist. "
                        f"Available worksheets: {available}."
                    )

                index = sheet_names.index(sheet_name)
                if index >= len(tables):
                    raise ValueError(
                        f"No data was loaded for worksheet '{sheet_name}'."
                    )

                selected = tables[index]
                if isinstance(selected, pd.DataFrame):
                    return selected.copy()

                raise ValueError(
                    f"Worksheet '{sheet_name}' contains no DataFrame."
                )

        elif sheet_name is not None:
            raise ValueError(
                "Worksheet selection is only supported for Excel workbooks."
            )

        for table in tables:
            if isinstance(table, pd.DataFrame):
                return table.copy()

    raise ValueError(
        "No pandas DataFrame was found in this document. "
        "Please load a CSV or Excel file containing tabular data."
    )


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Calculate exact row counts and descriptive statistics."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty DataFrame.")

    numeric_columns = df.select_dtypes(include="number").columns

    numeric_summary = {}

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        numeric_summary[column] = {
            "sum": float(series.sum()),
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            "count": int(series.count()),
        }

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "numeric_summary": numeric_summary,
        "missing_values": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
        },
    }


def group_by_summary(
    df: pd.DataFrame,
    group_by: str,
    metric: str,
    operation: str = "sum",
) -> pd.DataFrame:
    """Aggregate a numeric column by categories, such as revenue by region."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot analyze an empty DataFrame.")

    if group_by not in df.columns:
        raise ValueError(f"Grouping column '{group_by}' does not exist.")

    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' does not exist.")

    allowed_operations = {"sum", "mean", "min", "max", "count"}

    if operation not in allowed_operations:
        raise ValueError(
            f"Unsupported operation '{operation}'. "
            f"Choose one of: {', '.join(sorted(allowed_operations))}."
        )

    if operation != "count" and not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(
            f"Column '{metric}' must be numeric for '{operation}'."
        )

    result = (
        df.groupby(group_by, dropna=False)[metric]
        .agg(operation)
        .reset_index()
    )

    return result.sort_values(
        by=metric,
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)


def analyze_document(document: DocumentResult) -> dict:
    """Run structured analytics on a loaded CSV document."""

    df = get_dataframe(document)
    return analyze_dataframe(df)
