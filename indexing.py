"""
ingest()            – Re-index all PDFs found in `settings.data_dir`.
save_and_ingest()   – Save an uploaded PDF byte payload then index it.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import settings
from schemas import ChunkMetadata
from store import ensure_collection, get_vector_store


# Chunker protocol (allows swapping strategies )

class Chunker(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split page-level documents into chunk-level documents."""


# Internal helpers

def _make_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        # Priority order: paragraph → newline → sentence → word → character
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def _document_id(path: Path) -> str:
    """Derive a short stable ID from the filename + file size (16 hex chars)."""
    raw = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    """Compose a deterministic chunk identifier."""
    return f"{doc_id}:{page}:{index}"


# PDF loading 

def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF and attach normalised metadata to every page document.

    Page numbers are converted to 1-based integers.
    """
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    doc_id = _document_id(path)

    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1  
        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),  # populated only when available
        }

    return pages


# Discovery

def discover_pdfs(data_dir: Path | None = None) -> list[Path]:
    """Return a sorted list of all PDF files inside `data_dir`."""
    directory = data_dir or settings.data_dir
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


# Chunking

def build_chunks(
    pdf_paths: list[Path],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunker: Chunker | None = None,
) -> list[Document]:
    # 1. Load all pages
    page_docs: list[Document] = []
    for path in pdf_paths:
        logger.info("Loading PDF: {}", path.name)
        page_docs.extend(_load_pdf(path))

    if not page_docs:
        logger.warning("No page content extracted from provided PDF paths.")
        return []

    # 2. Split into chunks
    if chunker is None:
        chunks = _make_splitter(chunk_size, chunk_overlap).split_documents(page_docs)
    else:
        chunks = chunker.split_documents(page_docs)

    # 3. Attach validated metadata
    per_doc_counter: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1

        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id=_chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )
        chunk.metadata = meta.model_dump()

    logger.info("Produced {} chunks from {} page(s)", len(chunks), len(page_docs))
    return chunks


# Indexing

def index_chunks(
    chunks: list[Document],
    collection_name: str | None = None,
) -> int:
    
    if not chunks:
        return 0

    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"]))
        for c in chunks
    ]

    store = get_vector_store(collection_name=collection_name)
    store.add_documents(chunks, ids=ids)

    return len(chunks)


# High-level pipelines

def ingest(
    recreate: bool = False,
    collection_name: str | None = None,
    chunker: Chunker | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Full ingestion pipeline: discover → chunk → embed → index."""
    pdfs = discover_pdfs()
    if not pdfs:
        logger.warning("No PDF files found in {}", settings.data_dir)
        return 0

    ensure_collection(recreate=recreate, collection_name=collection_name)

    chunks = build_chunks(
        pdfs,
        chunker=chunker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        logger.warning("No chunks produced from {} PDF(s) – aborting.", len(pdfs))
        return 0

    count = index_chunks(chunks, collection_name=collection_name)
    logger.info("Ingested {} chunks from {} PDF(s).", count, len(pdfs))
    return count


def save_and_ingest(file_bytes: bytes, filename: str) -> dict[str, object]:
    """Save an uploaded PDF to `data_dir` then index it into Qdrant.

    This is the single entry point called by the API upload handler.
    """
    # Validate input
    if not filename:
        raise ValueError("Filename must not be empty.")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted.")
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    # Persist to disk 
    safe_name = Path(filename).name          # strip any directory traversal
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / safe_name
    dest.write_bytes(file_bytes)
    logger.info("Saved uploaded PDF to {}", dest)

    # Index
    ensure_collection(recreate=False)
    chunks = build_chunks([dest])

    if not chunks:
        logger.warning("No chunks produced for {}", safe_name)
        return {"filename": safe_name, "chunks_indexed": 0}

    count = index_chunks(chunks)
    logger.info("Indexed {} chunks from {}", count, safe_name)
    return {"filename": safe_name, "chunks_indexed": count}