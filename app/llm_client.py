
import ollama

from app.config import get_settings


def create_ollama_client() -> ollama.Client:
    """Create a client for the locally running Ollama server."""

    settings = get_settings()

    return ollama.Client(
        host=settings.ollama_base_url
    )


def check_ollama_health() -> dict:
    """Check Ollama connectivity and configured model availability."""

    settings = get_settings()
    client = create_ollama_client()

    response = client.list()

    available_models = [
        model.model for model in response.models
    ]

    if settings.ollama_model not in available_models:
        raise RuntimeError(
            f"Configured Ollama model '{settings.ollama_model}' "
            "is not installed."
        )

    return {
        "status": "healthy",
        "llm_provider": "ollama",
        "llm_model": settings.ollama_model,
        "model_available": True,
    }

def ask_llm(
    prompt: str,
    instructions: str = (
        "You are a careful document analysis assistant. "
        "Use only the supplied document evidence when answering "
        "document questions. If the evidence is insufficient, say so. "
        "Never invent facts or numerical results."
    ),
) -> str:
    """Generate a response using a locally running Ollama model."""

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    settings = get_settings()
    client = create_ollama_client()

    try:
        response = client.chat(
            model=settings.ollama_model,
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            think=False,
            keep_alive="30m",
            options={
                "temperature": 0.2,
                "num_ctx": 4096,
                "num_predict": 256,
            },
        )

        answer = response.message.content

        if not answer or not answer.strip():
            raise RuntimeError("Ollama returned an empty response.")

        return answer.strip()

    except Exception as exc:
        raise RuntimeError(
            "Could not connect to Ollama or generate a response. "
            "Check that Ollama is running and the model is installed."
        ) from exc


def generate_response(prompt: str) -> str:
    """Compatibility wrapper for the RAG question-answering pipeline."""

    return ask_llm(prompt)
