from __future__ import annotations
from openai import OpenAI

from cancer_immunology_reasoner.config import settings, BackendType


def get_client() -> OpenAI:
    """Get the appropriate OpenAI-compatible client based on settings."""
    kwargs = {}
    if settings.active_base_url:
        kwargs["base_url"] = settings.active_base_url
    if settings.active_api_key:
        kwargs["api_key"] = settings.active_api_key
    else:
        kwargs["api_key"] = "ollama"  # Ollama accepts any key
    return OpenAI(**kwargs)


def get_embedding_model() -> tuple[str, int]:
    """Get the appropriate embedding model name and dimension."""
    return settings.embedding_model, settings.embedding_dim


def get_reasoning_model() -> str:
    """Get the model to use for complex reasoning tasks."""
    return settings.reasoning_model


def get_fast_model() -> str:
    """Get the model to use for fast/simple tasks."""
    return settings.fast_model


def supports_json_mode() -> bool:
    return settings.supports_json_mode


def try_ollama_embedding(model: str, texts: list[str], client: OpenAI) -> list[list[float]] | None:
    """Try to get embeddings via Ollama's dedicated endpoint."""
    import requests
    try:
        base = settings.ollama_base_url.replace("/v1", "").replace("/v1/", "")
        resp = requests.post(
            f"{base}/api/embed",
            json={"model": model, "input": texts}
        )
        if resp.status_code == 200:
            data = resp.json()
            if "embeddings" in data:
                return data["embeddings"]
        return None
    except Exception:
        return None