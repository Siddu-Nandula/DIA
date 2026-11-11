import os

import ollama
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

client = ollama.Client(host=BASE_URL)


def generate_response(prompt: str) -> str:
    """Generate a response using the local Ollama model."""

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    response = client.generate(
        model=MODEL,
        prompt=prompt,
        stream=False,
    )

    return response["response"].strip()