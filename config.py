from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAG_",
        extra="ignore",
    )

    # Storage
    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage/qdrant")
    qdrant_collection: str = "pdf_chunks"

    # Chunking
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    # Retrieval
    top_k: int = Field(default=5, ge=1, le=64)

    # Embedding
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM (Gemini)
    llm_provider: Literal["gemini"] = "gemini"
    llm_model: str = "gemini-flash-lite-latest"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_new_tokens: int = Field(default=4096, ge=1, le=20000)

    # GEMINI_API_KEY
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")

    @model_validator(mode="after")
    def _validate_chunk_params(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()