"""
Local embeddings via Sentence-Transformers (all-MiniLM-L6-v2).
The model is loaded once and cached for the lifetime of the process.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model from HuggingFace."""
    return SentenceTransformer(settings.embedding_model)


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: 
        vectors = _load_model().encode(texts, normalize_embeddings=True)
        return self._to_list(vectors)

    def embed_query(self, text: str) -> list[float]: 
        vector = _load_model().encode([text], normalize_embeddings=True)[0]
        return self._to_float_list(vector)

    @staticmethod
    def _to_list(vectors) -> list[list[float]]: 
        tolist = getattr(vectors, "tolist", None)
        if callable(tolist):
            return tolist()
        return [[float(x) for x in row] for row in vectors]

    @staticmethod
    def _to_float_list(vector) -> list[float]:  
        tolist = getattr(vector, "tolist", None)
        if callable(tolist):
            return tolist()
        return [float(x) for x in vector]


def get_embeddings() -> Embeddings:
    return LocalEmbeddings()