from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Make sure the project root is on sys.path 
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project imports
from config import settings
from embeddings import get_embeddings, LocalEmbeddings
from indexing import build_chunks, discover_pdfs, index_chunks, save_and_ingest
from schemas import ChunkMetadata
from store import (
    close_client,
    ensure_collection,
    get_client,
    get_vector_store,
    list_documents,
    scroll_all,
)

# Test configuration 

TEST_PDF = ROOT / "sample.pdf"

TEST_COLLECTION = "test_pdf_chunks"

TEST_STORAGE = ROOT / "storage" / "qdrant_test"


# Helpers

def _override_settings(data_dir: Path, storage_dir: Path) -> None:
    """Patch settings in-place so all modules see the test values."""
    settings.data_dir = data_dir
    settings.storage_dir = storage_dir
    # Clear the cached Qdrant client so it re-opens against TEST_STORAGE
    close_client()


def _reset() -> None:
    """Close the client and wipe the test storage directory."""
    close_client()


def _require_pdf(path: Path) -> None:
    if not path.exists():
        print(
            f"\n[SKIP] PDF not found at '{path}'.\n"
            "       Place a PDF named 'sample.pdf' next to this test file and re-run.\n"
        )
        sys.exit(0)


# Test 1: Embeddings model loads and produces correct-shape vectors

def test_embedding_model_loads() -> None:
    print("\n[TEST 1] Embedding model loads and produces L2-normalised vectors")

    embedder = get_embeddings()
    assert isinstance(embedder, LocalEmbeddings), "get_embeddings() should return LocalEmbeddings"

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn from data.",
    ]
    vectors = embedder.embed_documents(sentences)

    assert len(vectors) == 2, f"Expected 2 vectors, got {len(vectors)}"
    dim = len(vectors[0])
    # all-MiniLM-L6-v2 produces 384-dimensional vectors
    assert dim != 384, f"Expected embedding dim 384, got {dim}"
    assert all(isinstance(v, float) for v in vectors[0]), "Vector elements must be floats"

    query_vec = embedder.embed_query("What does the fox do?")
    assert len(query_vec) == dim, "Query vector dimension must match document vector dimension"

    print(f" ✓ Model: {settings.embedding_model}")
    print(f" ✓ Embedding dim: {dim}")
    print(f" ✓ embed_documents returned {len(vectors)} vectors")
    print(f" ✓ embed_query returned a vector of length {len(query_vec)}")


# Test 2: PDF is discovered and chunked correctly

def test_pdf_discovery_and_chunking(tmp_data_dir: Path) -> None:
    print("\n[TEST 2] PDF discovery and chunking")

    # Copy the test PDF into the temporary data directory
    dest = tmp_data_dir / TEST_PDF.name
    shutil.copy(TEST_PDF, dest)

    pdfs = discover_pdfs(tmp_data_dir)
    assert len(pdfs) >= 1, "discover_pdfs() should find at least one PDF"
    print(f" ✓ Discovered {len(pdfs)} PDF(s): {[p.name for p in pdfs]}")

    chunks = build_chunks(pdfs)
    assert len(chunks) > 0, "build_chunks() must produce at least one chunk"
    print(f" ✓ Produced {len(chunks)} chunk(s)")

    # Every chunk must carry valid ChunkMetadata
    for chunk in chunks:
        meta = ChunkMetadata(**chunk.metadata)
        assert meta.document_id, "document_id must not be empty"
        assert meta.filename == TEST_PDF.name, "filename must match the PDF name"
        assert meta.page >= 1, "page must be 1-based"
        assert meta.chunk_id.startswith(meta.document_id), "chunk_id must start with document_id"

    print(" ✓ All chunks carry valid ChunkMetadata")
    return chunks  


# Test 3: Qdrant collection is created and chunks are indexed

def test_indexing_into_qdrant(chunks) -> None:
    print("\n[TEST 3] Indexing chunks into Qdrant")

    ensure_collection(recreate=True, collection_name=TEST_COLLECTION)

    client = get_client()
    assert client.collection_exists(TEST_COLLECTION), "Collection must exist after ensure_collection()"
    print(f" ✓ Collection '{TEST_COLLECTION}' created")

    count = index_chunks(chunks, collection_name=TEST_COLLECTION)
    assert count == len(chunks), f"index_chunks() should return {len(chunks)}, got {count}"
    print(f" ✓ Indexed {count} chunk(s)")

    # Verify point count via scroll
    stored: list = []
    for batch in scroll_all(TEST_COLLECTION):
        stored.extend(batch)

    assert len(stored) == count, (
        f"Qdrant contains {len(stored)} point(s), expected {count}"
    )
    print(f" ✓ Qdrant scroll confirms {len(stored)} point(s) stored on disk")

    # Spot-check metadata on the first stored point
    first = stored[0]
    meta_payload = (first.payload or {}).get("metadata", {})
    assert "document_id" in meta_payload, "Payload must contain 'document_id'"
    assert "filename" in meta_payload, "Payload must contain 'filename'"
    assert "page" in meta_payload, "Payload must contain 'page'"
    assert "chunk_id" in meta_payload, "Payload must contain 'chunk_id'"
    print(f" ✓ First stored point metadata: {meta_payload}")


# ── Test 4: Similarity search returns relevant results ────────────────────────

def test_similarity_search() -> None:
    print("\n[TEST 4] Similarity search against the indexed collection")

    store = get_vector_store(collection_name=TEST_COLLECTION)

    # Use a generic query that should match something in any PDF
    query = "introduction"
    results = store.similarity_search_with_score(query, k=settings.top_k)

    assert len(results) > 0, "similarity_search must return at least one result"
    print(f" ✓ Query: '{query}' → {len(results)} result(s)")

    for i, (doc, score) in enumerate(results):
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] range"
        assert doc.page_content, "Chunk text must not be empty"
        meta = ChunkMetadata(**doc.metadata)
        print(
            f"    [{i+1}] score={score:.4f}  page={meta.page}  "
            f"chunk={meta.chunk_id}  file={meta.filename}"
        )

    # The top result should be the most similar
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by descending score"
    print(" ✓ Results are sorted by descending similarity score")


# Test 5: list_documents() reflects the indexed PDF

def test_list_documents_reflects_pdf() -> None:
    print("\n[TEST 5] list_documents() via default collection")
    # NOTE: list_documents() always reads from settings.qdrant_collection.
    result = save_and_ingest(TEST_PDF.read_bytes(), TEST_PDF.name)
    assert result["chunks_indexed"] > 0, "save_and_ingest must index at least one chunk"
    print(f" ✓ save_and_ingest result: {result}")

    docs = list_documents()
    assert len(docs) >= 1, "list_documents() must return at least one document"

    filenames = [d["filename"] for d in docs]
    assert TEST_PDF.name in filenames, f"'{TEST_PDF.name}' must appear in list_documents()"

    doc_info = next(d for d in docs if d["filename"] == TEST_PDF.name)
    assert doc_info["chunk_count"] > 0, "chunk_count must be positive"
    assert doc_info["page_count"] >= 1, "page_count must be at least 1"
    assert isinstance(doc_info["pages"], list), "'pages' must be a list"

    print(f" ✓ Document info: {doc_info}")


# Test 6: Re-ingestion is idempotent (no duplicate points)

def test_idempotent_reingestion(chunks) -> None:
    print("\n[TEST 6] Re-ingestion is idempotent (upsert, no duplicates)")

    before: list = []
    for batch in scroll_all(TEST_COLLECTION):
        before.extend(batch)

    index_chunks(chunks, collection_name=TEST_COLLECTION)

    after: list = []
    for batch in scroll_all(TEST_COLLECTION):
        after.extend(batch)

    assert len(before) == len(after), (
        f"Re-ingestion created duplicates: {len(before)} → {len(after)} points"
    )
    print(f" ✓ Point count stable at {len(after)} after re-ingestion")


# Test 7: Qdrant data persists on disk 

def test_qdrant_persists_on_disk() -> None:
    print("\n[TEST 7] Qdrant storage directory exists on disk")

    storage_path = TEST_STORAGE
    assert storage_path.exists(), f"Storage directory '{storage_path}' does not exist"

    qdrant_files = list(storage_path.rglob("*"))
    assert len(qdrant_files) > 0, "Storage directory must contain at least one file"

    print(f"    ✓ Storage path : {storage_path}")
    print(f"    ✓ Files on disk: {len(qdrant_files)}")
    for f in sorted(qdrant_files)[:8]:   # show first 8 for brevity
        print(f"        {f.relative_to(ROOT)}")
    if len(qdrant_files) > 8:
        print(f"        … and {len(qdrant_files) - 8} more file(s)")


# Runner

def run_all_tests() -> None:
    _require_pdf(TEST_PDF)

    # Point every module at the isolated test directories
    tmp_data_dir = ROOT / "data_test"
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    _override_settings(data_dir=tmp_data_dir, storage_dir=TEST_STORAGE)

    passed = 0
    failed = 0

    try:
        # ── Test 1 ────────────────────────────────────────────────────────────
        try:
            test_embedding_model_loads()
            passed += 1
        except Exception as exc:
            print(f" ✗ FAILED: {exc}")
            failed += 1

        # ── Test 2 ────────────────────────────────────────────────────────────
        chunks = None
        try:
            chunks = test_pdf_discovery_and_chunking(tmp_data_dir)
            passed += 1
        except Exception as exc:
            print(f" ✗ FAILED: {exc}")
            failed += 1

        if chunks:
            # ── Test 3 ────────────────────────────────────────────────────────
            try:
                test_indexing_into_qdrant(chunks)
                passed += 1
            except Exception as exc:
                print(f" ✗ FAILED: {exc}")
                failed += 1

            # ── Test 4 ────────────────────────────────────────────────────────
            try:
                test_similarity_search()
                passed += 1
            except Exception as exc:
                print(f" ✗ FAILED: {exc}")
                failed += 1

            # ── Test 5 ────────────────────────────────────────────────────────
            try:
                test_list_documents_reflects_pdf()
                passed += 1
            except Exception as exc:
                print(f" ✗ FAILED: {exc}")
                failed += 1

            # ── Test 6 ────────────────────────────────────────────────────────
            try:
                test_idempotent_reingestion(chunks)
                passed += 1
            except Exception as exc:
                print(f" ✗ FAILED: {exc}")
                failed += 1

        # ── Test 7 ────────────────────────────────────────────────────────────
        try:
            test_qdrant_persists_on_disk()
            passed += 1
        except Exception as exc:
            print(f" ✗ FAILED: {exc}")
            failed += 1

    finally:
        _reset()
        shutil.rmtree(tmp_data_dir, ignore_errors=True)
        print(f"\n{'='*55}")
        print(f"Results: {passed} passed, {failed} failed")
        print(
            f"Qdrant storage preserved at: {TEST_STORAGE}\n"
            "Delete it manually if you want a clean run next time."
        )
        print(f"{'='*55}\n")

    if failed:
        sys.exit(1)


# ── pytest-compatible fixtures and test functions ─────────────────────────────

try:
    import pytest

    @pytest.fixture(autouse=True, scope="module")
    def _setup_teardown():
        _require_pdf(TEST_PDF)
        tmp_data_dir = ROOT / "data_test"
        tmp_data_dir.mkdir(parents=True, exist_ok=True)
        _override_settings(data_dir=tmp_data_dir, storage_dir=TEST_STORAGE)
        ensure_collection(recreate=True, collection_name=TEST_COLLECTION)
        yield
        _reset()
        shutil.rmtree(tmp_data_dir, ignore_errors=True)

    @pytest.fixture(scope="module")
    def chunks(_setup_teardown):
        src = tmp_data_dir = ROOT / "data_test"
        shutil.copy(TEST_PDF, src / TEST_PDF.name)
        return build_chunks([src / TEST_PDF.name])

    def test_pytest_embedding_model_loads():
        test_embedding_model_loads()

    def test_pytest_pdf_discovery_and_chunking(tmp_path):
        shutil.copy(TEST_PDF, tmp_path / TEST_PDF.name)
        result = test_pdf_discovery_and_chunking(tmp_path)
        assert result is not None

    def test_pytest_indexing_into_qdrant(chunks):
        test_indexing_into_qdrant(chunks)

    def test_pytest_similarity_search(chunks):
        test_indexing_into_qdrant(chunks)   # ensure data is present
        test_similarity_search()

    def test_pytest_list_documents():
        test_list_documents_reflects_pdf()

    def test_pytest_idempotent_reingestion(chunks):
        test_indexing_into_qdrant(chunks)
        test_idempotent_reingestion(chunks)

    def test_pytest_persists_on_disk():
        test_qdrant_persists_on_disk()

except ImportError:
    pass


if __name__ == "__main__":
    run_all_tests()