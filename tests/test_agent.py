
from types import SimpleNamespace
import pytest

from app.agent import run_agent
from app.document_loader import load_document


@pytest.fixture
def document():
    return load_document("data/sample_sales.csv")


@pytest.fixture
def excel_document():
    return load_document("data/sample_sales.xlsx")


def test_routes_analytics_question(document, monkeypatch):
    monkeypatch.setattr(
        "app.agent._choose_tool",
        lambda question: "analytics",
    )
    monkeypatch.setattr(
        "app.agent.execute_tool",
        lambda document, question: "Total revenue: 135,000.00",
    )

    result = run_agent(document, "What is the total revenue?")

    assert result.tool_used == "analytics"
    assert "135,000" in result.answer


def test_routes_chart_question(document, monkeypatch):
    monkeypatch.setattr(
        "app.agent._choose_tool",
        lambda question: "chart",
    )
    monkeypatch.setattr(
        "app.agent.execute_tool",
        lambda document, question: "Chart generated successfully.",
    )

    result = run_agent(document, "Create a revenue chart")

    assert result.tool_used == "chart"
    assert "Chart generated" in result.answer


def test_routes_document_question(document, monkeypatch):
    monkeypatch.setattr(
        "app.agent._choose_tool",
        lambda question: "document_qa",
    )
    monkeypatch.setattr(
        "app.agent.answer_question",
        lambda document, question: SimpleNamespace(
            answer="This document describes sales data."
        ),
    )

    result = run_agent(document, "Summarize this document")

    assert result.tool_used == "document_qa"
    assert "sales data" in result.answer


def test_rejects_empty_question(document):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        run_agent(document, "   ")


def test_routes_excel_analytics_question(excel_document, monkeypatch):
    monkeypatch.setattr(
        "app.agent._choose_tool",
        lambda question: "analytics",
    )
    monkeypatch.setattr(
        "app.agent.execute_tool",
        lambda document, question: (
            "Worksheet: Regions. Total rows: 3."
        ),
    )

    result = run_agent(
        excel_document,
        "How many rows are in the Regions worksheet?",
    )

    assert result.tool_used == "analytics"
    assert "Regions" in result.answer
    assert "3" in result.answer


def test_routes_excel_chart_question(excel_document, monkeypatch):
    monkeypatch.setattr(
        "app.agent._choose_tool",
        lambda question: "chart",
    )
    monkeypatch.setattr(
        "app.agent.execute_tool",
        lambda document, question: (
            "Chart generated successfully for the Regions worksheet."
        ),
    )

    result = run_agent(
        excel_document,
        "Create a chart for the Regions worksheet",
    )

    assert result.tool_used == "chart"
    assert "Regions" in result.answer
    assert "Chart generated" in result.answer
