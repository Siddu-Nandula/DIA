
import pytest
from app.agent_tools import _detect_time_frequency
from app.agent_tools import execute_tool, _detect_sheet_name
from app.document_loader import load_document


@pytest.fixture
def excel_document():
    return load_document("data/sample_sales.xlsx")


@pytest.fixture
def csv_document():
    return load_document("data/sample_sales.csv")


def test_detects_sales_worksheet(excel_document):
    result = _detect_sheet_name(
        excel_document,
        "Show revenue from the Sales worksheet",
    )
    assert result == "Sales"


def test_detects_worksheet_case_insensitively(excel_document):
    result = _detect_sheet_name(
        excel_document,
        "Create a chart using the sales worksheet",
    )
    assert result == "Sales"


def test_unknown_worksheet_uses_default(excel_document):
    result = _detect_sheet_name(
        excel_document,
        "Show revenue from the UnknownSheet worksheet",
    )
    assert result is None


def test_csv_has_no_worksheet_selection(csv_document):
    result = _detect_sheet_name(
        csv_document,
        "Analyze the Sales worksheet",
    )
    assert result is None


def test_empty_question_is_rejected(excel_document):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        execute_tool(excel_document, "   ")


def test_sales_worksheet_bar_chart(excel_document, monkeypatch):
    captured = {}

    def fake_generate_bar_chart(
        document,
        category,
        metric,
        operation="sum",
        filename="bar_chart.png",
        sheet_name=None,
    ):
        captured["category"] = category
        captured["metric"] = metric
        captured["operation"] = operation
        captured["sheet_name"] = sheet_name
        return "outputs/test_chart.png"

    monkeypatch.setattr(
        "app.agent_tools.generate_bar_chart",
        fake_generate_bar_chart,
    )

    result = execute_tool(
        excel_document,
        "Create a bar chart of revenue by product in the Sales worksheet",
    )

    assert captured["category"] == "product"
    assert captured["metric"] == "revenue"
    assert captured["operation"] == "sum"
    assert captured["sheet_name"] == "Sales"
    assert "Chart generated successfully" in result
    assert "outputs/test_chart.png" in result
def test_sales_worksheet_histogram(excel_document, monkeypatch):
    captured = {}

    def fake_generate_histogram(
        document,
        metric,
        filename,
        sheet_name,
    ):
        captured["metric"] = metric
        captured["sheet_name"] = sheet_name
        return "outputs/test_histogram.png"

    monkeypatch.setattr(
        "app.agent_tools.generate_histogram",
        fake_generate_histogram,
    )

    result = execute_tool(
        excel_document,
        "Generate a histogram for the revenue column in the Sales worksheet",
    )

    assert captured["metric"] == "revenue"
    assert captured["sheet_name"] == "Sales"
    assert "Histogram generated successfully" in result

def test_average_revenue_by_product(excel_document, monkeypatch):
    captured = {}

    def fake_generate_bar_chart(*args, **kwargs):
        captured.update(kwargs)
        return "outputs/test_average_chart.png"

    monkeypatch.setattr(
        "app.agent_tools.generate_bar_chart",
        fake_generate_bar_chart,
    )

    result = execute_tool(
        excel_document,
        "Create a chart of average revenue by product in the Sales worksheet",
    )

    assert captured["operation"] == "mean"
    assert captured["category"] == "product"
    assert captured["metric"] == "revenue"
    assert "Operation: mean" in result
def test_analytics_question_passes_worksheet(
    excel_document, monkeypatch
):
    captured = {}

    def fake_answer_analytics_question(
        document,
        question,
        sheet_name=None,
    ):
        captured["sheet_name"] = sheet_name
        return "Analytics answer"

    monkeypatch.setattr(
        "app.agent_tools.answer_analytics_question",
        fake_answer_analytics_question,
    )

    result = execute_tool(
        excel_document,
        "What is the total revenue in the Sales worksheet?",
    )

    assert captured["sheet_name"] == "Sales"
    assert result == "Analytics answer"


def test_regions_worksheet_does_not_use_sales_columns(
    excel_document, monkeypatch
):
    called = {}

    def fake_generate_bar_chart(*args, **kwargs):
        called["chart_generated"] = True
        return "outputs/test_chart.png"

    monkeypatch.setattr(
        "app.agent_tools.generate_bar_chart",
        fake_generate_bar_chart,
    )

    result = execute_tool(
        excel_document,
        "Create a chart of revenue by region in the Regions worksheet",
    )

    # The Regions worksheet does not contain a revenue column.
    assert "couldn't identify a numeric metric" in result
    assert "chart_generated" not in called



@pytest.mark.parametrize(
    "question, expected",
    [
        ("Show monthly revenue trends", "MS"),
        ("Show revenue month by month", "MS"),
        ("Show revenue by month", "MS"),
        ("Show weekly sales", "W"),
        ("Show daily orders", "D"),
        ("Show yearly profit", "YS"),
        ("Create a line chart", None),
    ],
)
def test_detect_time_frequency(question, expected):
    assert _detect_time_frequency(question) == expected


def test_compare_revenue_across_regions(csv_document, monkeypatch):
    captured = {}

    def fake_generate_bar_chart(
        document,
        category,
        metric,
        operation="sum",
        filename="bar_chart.png",
        sheet_name=None,
    ):
        captured["category"] = category
        captured["metric"] = metric
        captured["operation"] = operation
        return "outputs/test_comparison.png"

    monkeypatch.setattr(
        "app.agent_tools.generate_bar_chart",
        fake_generate_bar_chart,
    )

    result = execute_tool(
        csv_document,
        "Compare revenue across regions",
    )

    assert captured["category"] == "region"
    assert captured["metric"] == "revenue"
    assert captured["operation"] == "sum"
    assert "Chart generated successfully" in result


def test_show_distribution_of_units(csv_document, monkeypatch):
    captured = {}

    def fake_generate_histogram(
        document,
        metric,
        filename,
        sheet_name,
    ):
        captured["metric"] = metric
        return "outputs/test_units_histogram.png"

    monkeypatch.setattr(
        "app.agent_tools.generate_histogram",
        fake_generate_histogram,
    )

    result = execute_tool(
        csv_document,
        "Show the distribution of units",
    )
    def test_compare_revenue_across_regions(csv_document, monkeypatch):
        captured = {}

        def fake_generate_bar_chart(
            document,
            category,
            metric,
            operation="sum",
            filename="bar_chart.png",
            sheet_name=None,
        ):
            captured["category"] = category
            captured["metric"] = metric
            captured["operation"] = operation
            return "outputs/test_comparison.png"

        monkeypatch.setattr(
            "app.agent_tools.generate_bar_chart",
            fake_generate_bar_chart,
        )

        result = execute_tool(
            csv_document,
            "Compare revenue across regions",
        )

        print("EXECUTE_TOOL RESULT:", result)
        print("CAPTURED VALUES:", captured)

        assert captured["category"] == "region"
        assert captured["metric"] == "revenue"
        assert captured["operation"] == "sum"
        assert "Chart generated successfully" in result

        assert captured["metric"] == "units"
        assert "Histogram generated successfully" in result
        