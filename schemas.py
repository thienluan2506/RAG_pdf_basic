from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    document_id: str
    filename: str
    source: str
    page: int
    chunk_id: str
    section: str | None = None

class RetrievedChunk(BaseModel):
    """A chunk returned by a similarity search together with its score."""

    text: str
    score: float 
    metadata: ChunkMetadata

class Citation(BaseModel):
    """Source citation extracted from a retrieved chunk."""

    source_index: int           
    source_marker: str  
    filename: str
    page: int
    source_text: str | None = None   
    section: str | None = None
    chunk_id: str | None = None

class RagAnswer(BaseModel):
    """Final grounded answer returned to the caller."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)

class DocumentInfo(BaseModel):
    """Summary of one indexed PDF document."""

    filename: str
    document_id: str
    pages: list[int] = Field(default_factory=list)   # Sorted list of page numbers
    page_count: int
    chunk_count: int