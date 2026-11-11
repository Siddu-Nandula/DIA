
"""Automated tests for the Document Intelligence Agent API."""
from unittest.mock import patch

from app.agent import AgentResult
from io import BytesIO

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_rejects_unsupported_file():
    response = client.post(
        "/ask",
        files={
            "file": (
                "notes.txt",
                BytesIO(b"unsupported file"),
                "text/plain",
            )
        },
        data={"question": "Summarize this document"},
    )
    print("Response status:", response.status_code)
    print("Response body:", response.text)
    print(response.json())
    print("DEBUG STATUS:", response.status_code)
    print("DEBUG BODY:", response.text)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        files={
            "file": (
                "sample.csv",
                BytesIO(b"product,revenue\nPen,100\n"),
                "text/csv",
            )
        },
        data={"question": "   "},
    )

    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]
def test_ask_returns_agent_answer():
    csv_content = b"region,revenue\nSouth,98000\nNorth,25000\n"

    with patch(
        "app.api.run_agent",
        return_value=AgentResult(
            answer="Region with highest total revenue: South (98,000.00)",
            tool_used="analytics",
        ),
    ) as mock_agent:
        response = client.post(
            "/ask",
            files={
                "file": (
                    "sales.csv",
                    BytesIO(csv_content),
                    "text/csv",
                )
            },
            data={"question": "Which region generated highest revenue?"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Region with highest total revenue: South (98,000.00)"
    )
    assert response.json()["tool_used"] == "analytics"
    mock_agent.assert_called_once()
    
def test_chart_download_returns_png(monkeypatch, tmp_path):
    from app import api

    chart_file = tmp_path / "agent_chart.png"
    chart_file.write_bytes(b"fake-png-content")

    monkeypatch.setattr(api, "OUTPUTS_DIR", tmp_path)

    response = client.get("/charts/agent_chart.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"fake-png-content"


def test_chart_download_rejects_unknown_filename():
    response = client.get("/charts/secret.txt")

    assert response.status_code == 404
    assert response.json()["detail"] == "Chart not found."
def test_ask_rejects_oversized_file():
    from app.config import get_settings

    max_size = get_settings().max_upload_mb * 1024 * 1024
    oversized_content = b"x" * (max_size + 1)

    response = client.post(
        "/ask",
        files={
            "file": (
                "large.csv",
                BytesIO(oversized_content),
                "text/csv",
            )
        },
        data={"question": "Analyze this file"},
    )

    assert response.status_code == 413
    assert (
        f"Maximum size is {get_settings().max_upload_mb} MB"
        in response.json()["detail"]
    )