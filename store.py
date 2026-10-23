"""
All Qdrant state is kept on disk under `settings.storage_dir`
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings
from embeddings import get_embeddings
from pathlib import Path

# ── Constants ────

_SCROLL_PAGE_SIZE = 256

# Payload fields to index for fast filtered searches
INDEXED_PAYLOAD_FIELDS: dict[str, qmodels.PayloadSchemaType] = {
    "metadata.document_id": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.filename": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.page": qmodels.PayloadSchemaType.INTEGER,
}


# client
settings.storage_dir = (
    Path(__file__).parent.parent / "storage" / "qdrant"
).resolve()

@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return a cached local Qdrant client backed by on-disk storage."""
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.storage_dir))


def close_client() -> None:
    """Close and evict the cached client (useful in tests / CLI teardown)."""
    if get_client.cache_info().currsize == 0:
        return
    get_client().close()
    get_client.cache_clear()


# Collection management

def ensure_collection(
    recreate: bool = False,
    collection_name: str | None = None,
) -> None:
    """Create the Qdrant collection and payload indexes if they do not exist.

    Args:
        recreate: When True, drop and recreate the collection from scratch.
        collection_name: Override the default collection name from settings.
    """
    client = get_client()
    name = collection_name or settings.qdrant_collection

    exists = client.collection_exists(name)

    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if not exists:
        dim = len(get_embeddings().embed_query("dimension probe"))
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )

    # Create payload indexes only for fields that are missing or mismatched.
    payload_schema = client.get_collection(name).payload_schema or {}
    for field_name, expected_schema in INDEXED_PAYLOAD_FIELDS.items():
        existing = payload_schema.get(field_name)
        if existing is None:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=expected_schema,
            )
        elif getattr(existing, "data_type", None) != expected_schema:
            raise ValueError(
                f"Payload index '{field_name}' has unexpected schema "
                f"{getattr(existing, 'data_type', None)!r}; "
                f"expected {expected_schema!r}."
            )


# Scroll helper

def scroll_all(
    collection_name: str,
    scroll_filter: qmodels.Filter | None = None,
    with_payload: bool | list[str] = True,
    limit: int = _SCROLL_PAGE_SIZE,
) -> Iterator[list]:
    """Yield pages of Qdrant points (without vectors) until exhausted.

    Silently returns when the collection does not exist yet.
    """
    client = get_client()
    offset = None

    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=False,
            )
        except ValueError as exc:
            # Local Qdrant raises ValueError when the collection is absent.
            if "not found" in str(exc).lower():
                return
            raise

        yield points

        if next_offset is None:
            break
        offset = next_offset


# Vector store

def get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    """Return a LangChain QdrantVectorStore wired to the local client."""
    return QdrantVectorStore(
        client=get_client(),
        collection_name=collection_name or settings.qdrant_collection,
        embedding=get_embeddings(),
    )


# Document listing

def list_documents() -> list[dict[str, object]]:
    """Return one summary entry per indexed PDF.

    Each entry contains: filename, document_id, pages (sorted list),
    page_count, and chunk_count.
    """
    pages_map: dict[str, set[int]] = {}
    doc_id_map: dict[str, str] = {}
    count_map: dict[str, int] = {}

    for batch in scroll_all(settings.qdrant_collection, with_payload=["metadata"]):
        for point in batch:
            meta: dict = (point.payload or {}).get("metadata") or {}
            filename = meta.get("filename")
            document_id = meta.get("document_id")
            page = meta.get("page")

            if not filename or not document_id or not isinstance(page, int):
                continue

            fn = str(filename)
            doc_id_map.setdefault(fn, str(document_id))
            pages_map.setdefault(fn, set()).add(page)
            count_map[fn] = count_map.get(fn, 0) + 1

    return sorted(
        [
            {
                "filename": fn,
                "document_id": doc_id_map[fn],
                "pages": sorted(pages_map[fn]),
                "page_count": len(pages_map[fn]),
                "chunk_count": count_map[fn],
            }
            for fn in doc_id_map
        ],
        key=lambda d: str(d["filename"]),
    )