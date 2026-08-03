from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from enum import Enum


class BackendType(str, Enum):
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_backend: str = "ollama"
    openai_api_key: str = ""
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Ollama
    ollama_base_url: str = "http://localhost:11434/v1"

    # Model selection
    embedding_model: str = "nomic-embed-text"
    reasoning_model: str = "qwen3-coder:30b"
    fast_model: str = "gemma3:12b"
    embedding_dim: int = 768

    # Chunking
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Retrieval
    top_k_retrieval: int = 15
    max_graph_walk_depth: int = 3

    @property
    def backend_type(self) -> BackendType:
        return BackendType(self.llm_backend)

    @property
    def active_api_key(self) -> str:
        bt = self.backend_type
        if bt == BackendType.OPENAI:
            return self.openai_api_key
        elif bt == BackendType.GROQ:
            return self.groq_api_key
        return ""

    @property
    def active_base_url(self) -> str | None:
        bt = self.backend_type
        if bt == BackendType.OLLAMA:
            return self.ollama_base_url
        elif bt == BackendType.GROQ:
            return self.groq_base_url
        return None

    @property
    def supports_json_mode(self) -> bool:
        return self.backend_type in (BackendType.GROQ, BackendType.OPENAI)

    @property
    def supports_embeddings(self) -> bool:
        return self.backend_type in (BackendType.OPENAI, BackendType.OLLAMA)


settings = Settings()