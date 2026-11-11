
import pandas as pd
import pytest

from app.data_analytics import (
    analyze_dataframe,
    analyze_document,
    get_dataframe,
    group_by_summary,
)
from app.document_loader import DocumentResult, load_document


@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "product": ["Laptop", "Phone", "Tablet", "Monitor", "Phone"],
        "region": ["South", "North", "South", "West", "South"],
        "revenue": [50000, 25000, 18000, 12000, 30000],
        "units": [5, 10, 6, 4, 12],
    })


@pytest.fixture
def excel_document(tmp_path):
    """Create an Excel workbook with two worksheets for testing."""
    path = tmp_path / "sales.xlsx"

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({
            "product": ["Laptop", "Phone", "Tablet", "Monitor", "Phone"],
            "region": ["South", "North", "South", "West", "South"],
            "revenue": [50000, 25000, 18000, 12000, 30000],
            "units": [5, 10, 6, 4, 12],
        }).to_excel(writer, sheet_name="Sales", index=False)

        pd.DataFrame({
            "region": ["South", "North", "West"],
            "manager": ["Ravi", "Priya", "Arun"],
        }).to_excel(writer, sheet_name="Regions", index=False)

    return load_document(path)


def test_row_and_column_counts(sales_df):
    result = analyze_dataframe(sales_df)

    assert result["row_count"] == 5
    assert result["column_count"] == 4


def test_numeric_summary(sales_df):
    result = analyze_dataframe(sales_df)

    assert result["numeric_summary"]["revenue"]["sum"] == 135000.0
    assert result["numeric_summary"]["revenue"]["mean"] == 27000.0
    assert result["numeric_summary"]["units"]["sum"] == 37.0


def test_missing_values(sales_df):
    result = analyze_dataframe(sales_df)

    assert result["missing_values"]["product"] == 0
    assert result["missing_values"]["revenue"] == 0


def test_revenue_by_region(sales_df):
    result = group_by_summary(
        sales_df,
        group_by="region",
        metric="revenue",
        operation="sum",
    )

    assert result.iloc[0]["region"] == "South"
    assert result.iloc[0]["revenue"] == 98000


def test_average_revenue_by_region(sales_df):
    result = group_by_summary(
        sales_df,
        group_by="region",
        metric="revenue",
        operation="mean",
    )

    south = result[result["region"] == "South"].iloc[0]

    assert south["revenue"] == pytest.approx(98000 / 3)


def test_invalid_column_raises_error(sales_df):
    with pytest.raises(ValueError, match="does not exist"):
        group_by_summary(
            sales_df,
            group_by="invalid_region",
            metric="revenue",
        )


def test_empty_dataframe_raises_error():
    with pytest.raises(ValueError, match="empty DataFrame"):
        analyze_dataframe(pd.DataFrame())


def test_get_dataframe_selects_sales_worksheet(excel_document):
    df = get_dataframe(excel_document, sheet_name="Sales")

    assert len(df) == 5
    assert "revenue" in df.columns
    assert df["revenue"].sum() == 135000


def test_get_dataframe_selects_regions_worksheet(excel_document):
    df = get_dataframe(excel_document, sheet_name="Regions")

    assert len(df) == 3
    assert list(df.columns) == ["region", "manager"]


def test_unknown_worksheet_raises_error(excel_document):
    with pytest.raises(ValueError, match="does not exist"):
        get_dataframe(excel_document, sheet_name="Unknown")


def test_analyze_document_defaults_to_first_excel_worksheet(
    excel_document,
):
    result = analyze_document(excel_document)

    assert result["row_count"] == 5
    assert result["numeric_summary"]["revenue"]["sum"] == 135000.0
