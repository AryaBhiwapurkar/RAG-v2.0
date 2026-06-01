"""
core/chunker.py — Parent-child recursive text chunking.

WHY PARENT-CHILD:
  The tension in RAG chunking:
    - Small chunks → precise retrieval (specific, targeted match)
    - Large chunks → rich LLM context (enough text to answer well)
  Parent-child solves both:
    - Index small chunks (300 chars) for retrieval precision
    - When a small chunk matches, return its large parent (1200 chars) to LLM

WHY NOT fixed-size or semantic:
  - Fixed-size: ignores sentence/paragraph boundaries, cuts mid-thought
  - Semantic: too slow (requires embedding every potential split point)
  - Parent-child recursive: fast, respects structure, best tradeoff

DATA STRUCTURES:
  small_chunks: list[dict]  — what gets indexed
    {
      "chunk_id": "doc123_small_0",
      "parent_id": "doc123_large_0",
      "text": "...",           # 300 chars, used for retrieval
      "doc_id": "doc123",
      "page_num": 1,
    }

  large_chunks: dict[str, dict]  — keyed by parent_id, used at generation
    {
      "doc123_large_0": {
        "chunk_id": "doc123_large_0",
        "text": "...",         # 1200 chars, sent to LLM
        "doc_id": "doc123",
        "page_num": 1,
      }
    }
"""

import re
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Recursive character text splitter.
    Tries to split on paragraph breaks → newlines → spaces → characters.
    This respects natural text boundaries.

    Args:
        text: Input text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Characters to repeat at start of next chunk.

    Returns:
        List of text chunks.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks = []

    def _split(text: str, separators: list[str]) -> None:
        sep = separators[0] if separators else ""
        next_seps = separators[1:] if separators else []

        if len(text) <= chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return

        # Try to split on the current separator
        parts = text.split(sep) if sep else list(text)
        current = ""

        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Part itself is too large → recurse with finer separator
                if len(part) > chunk_size and next_seps:
                    _split(part, next_seps)
                elif part.strip():
                    chunks.append(part.strip()[:chunk_size])
                current = part.strip()[-overlap:] if overlap else part.strip()

        if current and current not in chunks:
            chunks.append(current)

    _split(text, separators)
    return [c for c in chunks if c.strip()]


def chunk_document(text: str, doc_id: str) -> tuple[list[dict], dict[str, dict]]:
    """
    Split document text into parent-child chunk pairs.

    Algorithm:
      1. Split text into large chunks (1200 chars) → these are parents
      2. For each large chunk, split into small chunks (300 chars) → children
      3. Each small chunk stores its parent_id for lookup at generation time

    Args:
        text: Full extracted text of the document.
        doc_id: Unique document identifier (used in chunk IDs).

    Returns:
        Tuple of:
          - small_chunks: list of dicts (what gets indexed in FAISS/BM25)
          - large_chunks: dict keyed by chunk_id (what gets sent to LLM)
    """
    # Step 1: Create large (parent) chunks
    large_texts = _split_text(
        text,
        chunk_size=settings.large_chunk_size,
        overlap=settings.large_chunk_overlap,
    )

    small_chunks: list[dict] = []
    large_chunks: dict[str, dict] = {}

    for large_idx, large_text in enumerate(large_texts):
        large_id = f"{doc_id}_large_{large_idx}"

        # Store the parent chunk
        large_chunks[large_id] = {
            "chunk_id": large_id,
            "text": large_text,
            "doc_id": doc_id,
            "char_count": len(large_text),
        }

        # Step 2: Split large chunk into small (child) chunks
        small_texts = _split_text(
            large_text,
            chunk_size=settings.small_chunk_size,
            overlap=settings.small_chunk_overlap,
        )

        for small_idx, small_text in enumerate(small_texts):
            small_id = f"{doc_id}_small_{large_idx}_{small_idx}"
            small_chunks.append({
                "chunk_id": small_id,
                "parent_id": large_id,   # ← key link: child → parent
                "text": small_text,
                "doc_id": doc_id,
                "char_count": len(small_text),
            })

    logger.info(
        f"Chunked doc '{doc_id}': "
        f"{len(large_chunks)} large chunks, {len(small_chunks)} small chunks"
    )
    return small_chunks, large_chunks
