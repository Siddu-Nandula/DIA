import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_model: str
    ollama_base_url: str
    max_upload_mb: int

    @classmethod
    def from_env(cls):
        model = os.getenv("OLLAMA_MODEL", "qwen3:4b").strip()
        if not model:
            raise ValueError("OLLAMA_MODEL cannot be empty")

        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ).strip()
        if not base_url:
            raise ValueError("OLLAMA_BASE_URL cannot be empty")

        raw_limit = os.getenv("MAX_UPLOAD_MB", "25")

        try:
            upload_limit = int(raw_limit)
        except (ValueError, TypeError):
            raise ValueError(
                "MAX_UPLOAD_MB must be a positive integer"
            ) from None

        if upload_limit <= 0:
            raise ValueError("MAX_UPLOAD_MB must be greater than zero")

        return cls(
            ollama_model=model,
            ollama_base_url=base_url,
            max_upload_mb=upload_limit,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()