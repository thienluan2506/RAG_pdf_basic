"""FastAPI server — bridge between Flutter frontend and the RAG backend.

Endpoints
---------
POST   /upload              Upload a PDF → index into Qdrant
POST   /chat                Ask a question → grounded RAG answer
GET    /documents           List all indexed PDFs
DELETE /documents/{doc_id}  Remove a PDF from Qdrant

Run
---
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Make sure project root is importable ─────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from indexing import save_and_ingest
from rag import answer
from store import get_client, list_documents, scroll_all
from config import settings

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PDF RAG API",
    description="Upload PDFs and ask questions grounded in their content.",
    version="1.0.0",
)

# Allow Flutter web / mobile to call this server from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    filename: str | None = None     # optional: restrict to one PDF
    top_k: int | None = None        # optional: override default top_k


class CitationOut(BaseModel):
    source_marker: str              # e.g. "S1"
    filename: str
    page: int
    section: str | None = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationOut]


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str


class DocumentOut(BaseModel):
    filename: str
    document_id: str
    page_count: int
    chunk_count: int


class DeleteResponse(BaseModel):
    document_id: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Quick liveness check."""
    return {"status": "ok", "service": "PDF RAG API"}


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_pdf(file: UploadFile = File(...)):
    """Receive a PDF from Flutter, save it, and index it into Qdrant.

    Flutter side:
        MultipartRequest  POST /upload
        field name: "file"
        content-type: application/pdf
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Index ─────────────────────────────────────────────────────────────────
    try:
        result = save_and_ingest(file_bytes, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return UploadResponse(
        filename=str(result["filename"]),
        chunks_indexed=int(result["chunks_indexed"]),
        message=f"Đã index {result['chunks_indexed']} chunks từ '{result['filename']}'.",
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    """Ask a question and receive a grounded answer with citations.

    Flutter side:
        POST /chat
        Body: { "question": "...", "filename": "report.pdf" }
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    filters = {"filename": req.filename} if req.filename else None

    try:
        result = answer(
            question=req.question,
            k=req.top_k,
            filters=filters,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    citations_out = [
        CitationOut(
            source_marker=c.source_marker,
            filename=c.filename,
            page=c.page,
            section=c.section,
        )
        for c in result.citations
    ]

    return ChatResponse(
        question=result.question,
        answer=result.answer,
        citations=citations_out,
    )


@app.get("/documents", response_model=list[DocumentOut], tags=["Documents"])
def get_documents():
    """Return a list of all indexed PDFs with metadata.

    Flutter side:
        GET /documents
        Returns: [ { filename, document_id, page_count, chunk_count }, ... ]
    """
    try:
        docs = list_documents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return [
        DocumentOut(
            filename=str(d["filename"]),
            document_id=str(d["document_id"]),
            page_count=int(d["page_count"]),
            chunk_count=int(d["chunk_count"]),
        )
        for d in docs
    ]


@app.delete("/documents/{document_id}", response_model=DeleteResponse, tags=["Documents"])
def delete_document(document_id: str):
    """Delete all Qdrant points belonging to a document_id.

    Flutter side:
        DELETE /documents/<document_id>
    """
    from qdrant_client.http import models as qmodels

    client = get_client()
    collection = settings.qdrant_collection

    if not client.collection_exists(collection):
        raise HTTPException(status_code=404, detail="Collection does not exist.")

    # Collect point IDs that belong to this document
    point_ids: list = []
    scroll_filter = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="metadata.document_id",
                match=qmodels.MatchValue(value=document_id),
            )
        ]
    )
    for batch in scroll_all(collection, scroll_filter=scroll_filter, with_payload=False):
        point_ids.extend(p.id for p in batch)

    if not point_ids:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for document_id '{document_id}'.",
        )

    client.delete(
        collection_name=collection,
        points_selector=qmodels.PointIdsList(points=point_ids),
    )

    return DeleteResponse(
        document_id=document_id,
        message=f"Đã xóa {len(point_ids)} chunks của document '{document_id}'.",
    )
