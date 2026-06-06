"""
pipeline/ingest.py — Full ingestion pipeline: PDF → indexed, searchable chunks.

PHASE 3 CHANGE:
  - Doc registry now uses SQLite (storage/registry.py) instead of flat JSON.
  - All registry logic (load, save, dedup) moved to storage/registry.py.
  - This file is now clean: ingest logic only, no JSON I/O.
  - Public interface unchanged — routes.py and api/routes.py call ingest_document()
    and get_registry() exactly as before.

PHASE 1 (still active):
  - Dedup check: if filename already exists with status="ready", skip re-ingestion.
  - Stale "processing" / "failed" entries cleaned up (handled inside find_existing()).
"""

import pdfplumber
import uuid
import logging
import time
import shutil
from datetime import datetime
from pathlib import Path
from config.settings import settings
from core.chunker import chunk_document
from core.embedder import embed_batch
from storage.vector_store import save_index, build_bm25_index, index_exists
from storage.registry import (
    get_registry,
    update_registry,
    find_existing,
    delete_registry_entry,
)

logger = logging.getLogger(__name__)


# ── PDF VALIDATION ─────────────────────────────────────────────────────────────

def validate_pdf(path: str | Path) -> tuple[bool, str]:
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
            "V2 supports text-based PDFs only."
        )
    if chars_per_page < settings.pdf_min_chars_per_page:
        return False, (
            f"Too little text per page (~{chars_per_page:.0f} chars). "
            "PDF may be image-heavy."
        )
    if total_pages > settings.pdf_max_pages:
        return False, (
            f"PDF has {total_pages} pages (max {settings.pdf_max_pages}). "
            "Split into smaller sections for best results."
        )

    return True, f"Valid: {total_pages} pages, ~{int(chars_per_page)} chars/page"


# ── TEXT EXTRACTION ────────────────────────────────────────────────────────────

def extract_text(path: str | Path) -> tuple[str, int]:
    with pdfplumber.open(path) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        full_text = "\n\n".join(pages)
        return full_text, len(pdf.pages)


# ── MAIN INGEST ────────────────────────────────────────────────────────────────

def ingest_document(file_path: str | Path, filename: str) -> dict:
    """
    Full ingestion pipeline for a single PDF.
    Skips if filename is already ingested and ready (dedup).
    """
    # ── Dedup check ────────────────────────────────────────────────────────
    existing_id = find_existing(filename)
    if existing_id:
        logger.info(f"[INGEST] '{filename}' already ready as doc_id='{existing_id}' — skipping.")
        return {"doc_id": existing_id, "status": "ready", "skipped": True}

    t_start = time.time()
    doc_id = str(uuid.uuid4())[:8]

    logger.info(f"Starting ingestion: '{filename}' → doc_id='{doc_id}'")

    update_registry(doc_id, {
        "doc_id": doc_id,
        "filename": filename,
        "status": "processing",
        "ingested_at": datetime.utcnow().isoformat(),
    })

    try:
        # Step 1: Validate
        valid, msg = validate_pdf(file_path)
        if not valid:
            update_registry(doc_id, {"status": "failed", "error": msg})
            logger.error(f"Validation failed for '{filename}': {msg}")
            return {"doc_id": doc_id, "status": "failed", "error": msg}
        logger.info(f"Validation passed: {msg}")

        # Step 2: Extract
        full_text, page_count = extract_text(file_path)
        logger.info(f"Extracted {len(full_text):,} chars from {page_count} pages")

        # Step 3: Chunk
        small_chunks, large_chunks = chunk_document(full_text, doc_id)
        logger.info(f"Chunked: {len(small_chunks)} small chunks, {len(large_chunks)} large chunks")

        # Step 4: Embed
        texts = [c["text"] for c in small_chunks]
        vectors = embed_batch(texts)

        # Step 5: BM25
        bm25_index = build_bm25_index(small_chunks)

        # Step 6: Save
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

        update_registry(doc_id, result)
        logger.info(f"Ingestion complete for '{filename}' (doc_id='{doc_id}') in {elapsed:.1f}s")
        return result

    except Exception as e:
        elapsed = time.time() - t_start
        update_registry(doc_id, {"status": "failed", "error": str(e)})
        logger.exception(f"Ingestion failed for '{filename}' after {elapsed:.1f}s: {e}")
        return {"doc_id": doc_id, "status": "failed", "error": str(e)}