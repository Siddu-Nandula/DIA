
import pandas as pd
import pytest
import matplotlib.axes

from app import chart_generator
from app.document_loader import load_document
from app.chart_generator import (
    generate_bar_chart,
    generate_histogram,
    generate_line_chart,
)


@pytest.fixture
def excel_document():
    return load_document("data/sample_sales.xlsx")


@pytest.fixture
def output_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(chart_generator, "OUTPUT_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def time_series_data():
    return pd.DataFrame({
        "date": [
            "2023-12-31",
            "2024-01-01",
            "2024-01-03",
            "2024-01-08",
            "2025-01-02",
        ],
        "revenue": [100, 200, 300, 400, 500],
    })


@pytest.fixture
def capture_plot(monkeypatch):
    """Capture the actual X and Y values passed to Matplotlib."""

    captured = {}

    def fake_plot(self, x, y, *args, **kwargs):
        captured["x"] = list(x)
        captured["y"] = list(y)
        return []

    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "plot",
        fake_plot,
    )

    return captured


# ---------------------------------------------------------
# BAR CHART TESTS
# ---------------------------------------------------------

def test_generate_bar_chart_for_sales_worksheet(
    excel_document, output_directory
):
    result = generate_bar_chart(
        excel_document,
        category="product",
        metric="revenue",
        sheet_name="Sales",
    )

    output_path = output_directory / "bar_chart.png"

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_bar_chart_rejects_missing_metric(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="Metric column 'revenue'"):
        generate_bar_chart(
            excel_document,
            category="region",
            metric="revenue",
            sheet_name="Regions",
        )


def test_bar_chart_rejects_unknown_worksheet(
    excel_document, output_directory
):
    with pytest.raises(ValueError):
        generate_bar_chart(
            excel_document,
            category="region",
            metric="revenue",
            sheet_name="UnknownSheet",
        )


def test_bar_chart_rejects_non_numeric_metric(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="must be numeric"):
        generate_bar_chart(
            excel_document,
            category="region",
            metric="product",
            sheet_name="Sales",
        )


def test_bar_chart_rejects_invalid_operation(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="Operation must be"):
        generate_bar_chart(
            excel_document,
            category="region",
            metric="revenue",
            operation="median",
            sheet_name="Sales",
        )


def test_chart_rejects_invalid_filename(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="simple .png filename"):
        generate_bar_chart(
            excel_document,
            category="region",
            metric="revenue",
            filename="../unsafe.png",
            sheet_name="Sales",
        )


# ---------------------------------------------------------
# HISTOGRAM TESTS
# ---------------------------------------------------------

def test_generate_histogram_for_revenue(
    excel_document, output_directory
):
    result = generate_histogram(
        excel_document,
        metric="revenue",
        sheet_name="Sales",
    )

    output_path = output_directory / "histogram.png"

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_histogram_for_units(
    excel_document, output_directory
):
    result = generate_histogram(
        excel_document,
        metric="units",
        filename="units_histogram.png",
        sheet_name="Sales",
    )

    output_path = output_directory / "units_histogram.png"

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_histogram_rejects_unknown_worksheet(
    excel_document, output_directory
):
    with pytest.raises(ValueError):
        generate_histogram(
            excel_document,
            metric="revenue",
            sheet_name="UnknownSheet",
        )


def test_histogram_rejects_missing_column(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="does not exist"):
        generate_histogram(
            excel_document,
            metric="nonexistent_column",
            sheet_name="Sales",
        )


def test_histogram_rejects_non_numeric_column(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="must be numeric"):
        generate_histogram(
            excel_document,
            metric="product",
            sheet_name="Sales",
        )


# ---------------------------------------------------------
# LINE CHART TESTS
# ---------------------------------------------------------

def test_generate_line_chart_for_sales_worksheet(
    excel_document, output_directory
):
    result = generate_line_chart(
        excel_document,
        x_axis="product",
        metric="revenue",
        sheet_name="Sales",
    )

    output_path = output_directory / "line_chart.png"

    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_line_chart_rejects_missing_x_axis(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="X-axis column"):
        generate_line_chart(
            excel_document,
            x_axis="nonexistent_column",
            metric="revenue",
            sheet_name="Sales",
        )


def test_line_chart_rejects_missing_metric(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="Metric column"):
        generate_line_chart(
            excel_document,
            x_axis="product",
            metric="nonexistent_column",
            sheet_name="Sales",
        )


def test_line_chart_rejects_non_numeric_metric(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="must be numeric"):
        generate_line_chart(
            excel_document,
            x_axis="region",
            metric="product",
            sheet_name="Sales",
        )


def test_line_chart_rejects_invalid_filename(
    excel_document, output_directory
):
    with pytest.raises(ValueError, match="simple .png filename"):
        generate_line_chart(
            excel_document,
            x_axis="product",
            metric="revenue",
            filename="../unsafe.png",
            sheet_name="Sales",
        )


# ---------------------------------------------------------
# TIME AGGREGATION TESTS
# ---------------------------------------------------------

@pytest.mark.parametrize(
    ("frequency", "expected_values"),
    [
        ("MS", [100, 900, 500]),
        ("W", [100, 500, 400, 500]),
        ("D", [100, 200, 300, 400, 500]),
        ("YS", [100, 900, 500]),
    ],
)
def test_line_chart_time_aggregation(
    excel_document,
    output_directory,
    time_series_data,
    capture_plot,
    monkeypatch,
    frequency,
    expected_values,
):
    monkeypatch.setattr(
        chart_generator,
        "get_dataframe",
        lambda document, sheet_name=None: time_series_data.copy(),
    )

    generate_line_chart(
        excel_document,
        x_axis="date",
        metric="revenue",
        frequency=frequency,
    )

    assert capture_plot["y"] == expected_values


def test_monthly_mean_aggregation(
    excel_document,
    output_directory,
    time_series_data,
    capture_plot,
    monkeypatch,
):
    monkeypatch.setattr(
        chart_generator,
        "get_dataframe",
        lambda document, sheet_name=None: time_series_data.copy(),
    )

    generate_line_chart(
        excel_document,
        x_axis="date",
        metric="revenue",
        frequency="MS",
        operation="mean",
    )

    assert capture_plot["y"] == [100, 300, 500]


def test_monthly_count_aggregation(
    excel_document,
    output_directory,
    time_series_data,
    capture_plot,
    monkeypatch,
):
    monkeypatch.setattr(
        chart_generator,
        "get_dataframe",
        lambda document, sheet_name=None: time_series_data.copy(),
    )

    generate_line_chart(
        excel_document,
        x_axis="date",
        metric="revenue",
        frequency="MS",
        operation="count",
    )

    assert capture_plot["y"] == [1, 3, 1]


@pytest.mark.parametrize(
    "frequency",
    ["invalid", "M", "Q"],
)
def test_line_chart_rejects_unsupported_frequency(
    excel_document,
    output_directory,
    frequency,
):
    with pytest.raises(ValueError, match="Frequency must be"):
        generate_line_chart(
            excel_document,
            x_axis="product",
            metric="revenue",
            frequency=frequency,
        )


def test_line_chart_rejects_invalid_aggregation_operation(
    excel_document,
    output_directory,
):
    with pytest.raises(ValueError, match="Operation must be"):
        generate_line_chart(
            excel_document,
            x_axis="product",
            metric="revenue",
            frequency="MS",
            operation="median",
        )
