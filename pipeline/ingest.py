"""
pipeline/ingest.py — Full ingestion pipeline: PDF → indexed, searchable chunks.

FLOW (per document, run once):
  PDF file
    ↓ pdfplumber: extract text + validate
    ↓ chunker.py: parent-child split
    ↓ embedder.py: embed small chunks → 768-dim vectors
    ↓ vector_store.py: save FAISS index + BM25 + chunks to disk
    ↓ doc_registry.json: record metadata (id, filename, pages, chunk count)

WHY pdfplumber (not PyPDF2/pypdf):
  pdfplumber preserves text structure better (spacing, table layout).
  For legal/research docs with tables, it extracts table cells as text.
  pypdf is simpler but loses structure on complex PDFs.

WHY ASYNC background task:
  Large PDFs (50+ pages) take 10-30 seconds to embed.
  Running synchronously blocks the UI and timeouts FastAPI requests.
  FastAPI's BackgroundTasks runs ingest after sending the HTTP response.
  The UI polls GET /documents to know when ingestion is complete.

DOCUMENT REGISTRY (doc_registry.json):
  Single JSON file tracking all ingested documents:
  {
    "doc_id": {
      "doc_id": "abc123",
      "filename": "contract.pdf",
      "pages": 42,
      "small_chunks": 310,
      "large_chunks": 78,
      "status": "ready",  # "processing" | "ready" | "failed"
      "ingested_at": "2026-05-30T12:00:00"
    }
  }
"""

import pdfplumber
import json
import uuid
import logging
import time
from datetime import datetime
from pathlib import Path
from config.settings import settings
from core.chunker import chunk_document
from core.embedder import embed_batch
from storage.vector_store import save_index, build_bm25_index, index_exists

logger = logging.getLogger(__name__)


# ── PDF VALIDATION ─────────────────────────────────────────────────────────────

def validate_pdf(path: str | Path) -> tuple[bool, str]:
    """
    Check if a PDF is text-based and within scope.

    V2 explicitly scopes to text-based PDFs only. Scanned/image PDFs
    need OCR (pytesseract + pdf2image) — deferred to V3.

    Returns:
        (is_valid: bool, message: str)
    """
    try:
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            total_chars = sum(
                len(page.extract_text() or "") for page in pdf.pages
            )
    except Exception as e:
        return False, f"Could not open PDF: {e}"

    chars_per_page = total_chars / max(total_pages, 1)

    if total_chars < settings.pdf_min_total_chars:
        return False, (
            "PDF appears to be scanned or image-only (very little text found). "
            "V2 supports text-based PDFs only. OCR support is planned for V3."
        )
    if chars_per_page < settings.pdf_min_chars_per_page:
        return False, (
            f"Too little text per page (~{chars_per_page:.0f} chars). "
            "PDF may be image-heavy. V2 supports text-based PDFs only."
        )
    if total_pages > settings.pdf_max_pages:
        return False, (
            f"PDF has {total_pages} pages (max {settings.pdf_max_pages}). "
            "Split into smaller sections for best results."
        )

    return True, f"Valid: {total_pages} pages, ~{int(chars_per_page)} chars/page"


# ── TEXT EXTRACTION ────────────────────────────────────────────────────────────

def extract_text(path: str | Path) -> tuple[str, int]:
    """
    Extract all text from a PDF using pdfplumber.

    Returns:
        (full_text: str, page_count: int)
    """
    with pdfplumber.open(path) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        full_text = "\n\n".join(pages)
        return full_text, len(pdf.pages)


# ── DOCUMENT REGISTRY ─────────────────────────────────────────────────────────

def _load_registry() -> dict:
    """Load doc registry from disk, return empty dict if not exists."""
    if settings.doc_registry_path.exists():
        with open(settings.doc_registry_path, "r") as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict) -> None:
    """Persist doc registry to disk."""
    settings.doc_registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.doc_registry_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def get_registry() -> dict:
    """Public accessor for the document registry."""
    return _load_registry()


def _update_registry(doc_id: str, update: dict) -> None:
    """Update a single document's entry in the registry."""
    registry = _load_registry()
    if doc_id not in registry:
        registry[doc_id] = {}
    registry[doc_id].update(update)
    _save_registry(registry)


# ── MAIN INGEST ────────────────────────────────────────────────────────────────

def ingest_document(file_path: str | Path, filename: str) -> dict:
    """
    Full ingestion pipeline for a single PDF.

    This is the main entry point called by the FastAPI background task.

    Args:
        file_path: Path to the uploaded PDF file.
        filename: Original filename (for display in UI).

    Returns:
        Result dict with doc_id, status, chunk counts, timing.
    """
    t_start = time.time()
    doc_id = str(uuid.uuid4())[:8]  # Short unique ID (e.g. "a1b2c3d4")

    logger.info(f"Starting ingestion: '{filename}' → doc_id='{doc_id}'")

    # Register as "processing" immediately so UI can show progress
    _update_registry(doc_id, {
        "doc_id": doc_id,
        "filename": filename,
        "status": "processing",
        "ingested_at": datetime.utcnow().isoformat(),
    })

    try:
        # ── Step 1: Validate ──────────────────────────────────────────────
        valid, msg = validate_pdf(file_path)
        if not valid:
            _update_registry(doc_id, {"status": "failed", "error": msg})
            logger.error(f"Validation failed for '{filename}': {msg}")
            return {"doc_id": doc_id, "status": "failed", "error": msg}

        logger.info(f"Validation passed: {msg}")

        # ── Step 2: Extract text ──────────────────────────────────────────
        full_text, page_count = extract_text(file_path)
        logger.info(f"Extracted {len(full_text):,} chars from {page_count} pages")

        # ── Step 3: Chunk ─────────────────────────────────────────────────
        small_chunks, large_chunks = chunk_document(full_text, doc_id)
        logger.info(
            f"Chunked: {len(small_chunks)} small chunks, "
            f"{len(large_chunks)} large chunks"
        )

        # ── Step 4: Embed small chunks ────────────────────────────────────
        texts = [c["text"] for c in small_chunks]
        vectors = embed_batch(texts)  # Shape: (N, 768)

        # ── Step 5: Build BM25 index ──────────────────────────────────────
        bm25_index = build_bm25_index(small_chunks)

        # ── Step 6: Save all indexes to disk ──────────────────────────────
        save_index(doc_id, vectors, small_chunks, large_chunks, bm25_index)

        elapsed = time.time() - t_start
        result = {
            "doc_id": doc_id,
            "filename": filename,
            "status": "ready",
            "pages": page_count,
            "small_chunks": len(small_chunks),
            "large_chunks": len(large_chunks),
            "total_chars": len(full_text),
            "ingestion_time_s": round(elapsed, 1),
        }

        _update_registry(doc_id, result)
        logger.info(
            f"Ingestion complete for '{filename}' (doc_id='{doc_id}') "
            f"in {elapsed:.1f}s"
        )
        return result

    except Exception as e:
        elapsed = time.time() - t_start
        error_msg = str(e)
        _update_registry(doc_id, {"status": "failed", "error": error_msg})
        logger.exception(f"Ingestion failed for '{filename}' after {elapsed:.1f}s: {e}")
        return {"doc_id": doc_id, "status": "failed", "error": error_msg}
