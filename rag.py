from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from config import settings
from filters import filters_to_qdrant
from llm import invoke_llm
from schemas import ChunkMetadata, Citation, RagAnswer, RetrievedChunk
from store import get_vector_store

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"

_NO_CONTEXT_REPLY = (
    "Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
)


# Jinja2 environment (cached)

@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(template_name: str, **context: object) -> str:
    """Render a Jinja2 template from the prompts/ directory"""
    return _jinja_env().get_template(template_name).render(**context)

# Retrieval

def retrieve(
    query: str,
    k: int | None = None,
    filters: dict[str, object] | None = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    """Run a similarity search against Qdrant and return the top-k chunks."""
    store = get_vector_store(collection_name=collection_name)

    hits = store.similarity_search_with_score(
        query=query,
        k=k or settings.top_k,
        filter=filters_to_qdrant(filters),
    )

    return [
        RetrievedChunk(
            text=doc.page_content,
            score=float(score),
            metadata=ChunkMetadata(**doc.metadata),
        )
        for doc, score in hits
    ]

# Citation helpers

def format_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Build a `Citation` for each chunk, using 1-based [S1], [S2], … markers."""
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            source_text=c.text.strip(),
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]

# Full RAG pipeline

def answer(
    question: str,
    k: int | None = None,
    filters: dict[str, object] | None = None,
    collection_name: str | None = None,
) -> RagAnswer:
    """End-to-end RAG: retrieve relevant chunks → build prompt → call LLM.

    Returns:
        `RagAnswer` containing the answer text, citations, and raw chunks.
        When no relevant context exists, returns a fallback message
    """
    # Step 1: retrieve
    chunks = retrieve(question, k=k, filters=filters, collection_name=collection_name)

    if not chunks:
        return RagAnswer(
            question=question,
            answer=_NO_CONTEXT_REPLY,
            citations=[],
            chunks=[],
        )

    # Step 2: render prompt
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=chunks)

    # Step 3: call LLM
    raw_answer = invoke_llm(prompt)

    # Step 4: assemble result
    return RagAnswer(
        question=question,
        answer=raw_answer.strip(),
        citations=format_citations(chunks),
        chunks=chunks,
    )
