import pytest

from app.config import Settings


def test_loads_settings_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )
    monkeypatch.setenv("MAX_UPLOAD_MB", "30")

    settings = Settings.from_env()

    assert settings.ollama_model == "qwen3:4b"
    assert (
        settings.ollama_base_url
        == "http://localhost:11434"
    )
    assert settings.max_upload_mb == 30


def test_missing_api_key_is_not_required(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv("MAX_UPLOAD_MB", "20")

    settings = Settings.from_env()

    assert settings.ollama_model == "qwen3:4b"


def test_rejects_invalid_upload_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "invalid")

    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_env()


def test_rejects_zero_upload_limit(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        Settings.from_env()


def test_rejects_empty_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", " ")

    with pytest.raises(ValueError, match="cannot be empty"):
        Settings.from_env()