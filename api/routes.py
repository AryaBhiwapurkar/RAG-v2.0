"""
api/routes.py — FastAPI router: /ingest, /query, /documents, /metrics.

WHY FASTAPI (not Flask or Gradio-only):
  - Async support (async def) → non-blocking I/O → handles concurrent requests
  - BackgroundTasks → ingest runs after HTTP response is sent → no UI blocking
  - Pydantic validation → automatic 422 errors for bad requests
  - Auto OpenAPI docs at /docs → free, no extra work

ENDPOINTS:
  POST /ingest    — Upload a PDF, start background ingestion
  POST /query     — Ask a question, get an answer
  GET  /documents — List all ingested documents and their status
  GET  /metrics   — Latency + cache stats

ASYNC INGESTION FLOW:
  1. Client POSTs PDF → server saves file, returns 202 with doc_id immediately
  2. Ingestion runs in background (chunking, embedding, indexing)
  3. Client polls GET /documents to check when status = "ready"
  4. Once ready, client can query
"""

import shutil
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from api.models import (
    QueryRequest, QueryResponse, IngestResponse,
    DocumentListResponse, DocumentInfo, MetricsResponse, TokenUsage
)
from pipeline.ingest import ingest_document, get_registry
from pipeline.query import run_query
from storage.cache import get_cache
from evaluation.latency_tracker import get_tracker
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ── POST /ingest ───────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF and start background ingestion.

    Returns immediately with doc_id and status="processing".
    Poll GET /documents to check when status="ready".
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to disk
    save_path = settings.uploads_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Start ingestion in background (non-blocking)
    # ingest_document will update the registry as it progresses
    background_tasks.add_task(ingest_document, save_path, file.filename)

    logger.info(f"Queued ingestion for '{file.filename}'")

    return IngestResponse(
        doc_id="pending",   # Will be assigned by ingest_document
        filename=file.filename,
        status="processing",
        message=(
            f"'{file.filename}' is being processed. "
            "Poll GET /documents to check when status='ready'."
        ),
    )


# ── POST /query ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question against ingested documents.

    If doc_ids is null, searches all ready documents.
    Returns answer with latency, source, and faithfulness metadata.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = run_query(
        question=request.question,
        doc_ids=request.doc_ids,
    )

    # Build confidence message for UI display
    flag = result.get("faithfulness_flag", "high")
    confidence_messages = {
        "high": None,  # No message needed — show answer normally
        "medium": "Confidence: Medium — please verify this answer against the source document.",
        "low": "Low confidence — I couldn't find a reliable answer. Please check the document directly.",
    }

    token_usage = result.get("token_usage", {})

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        cache_hit=result.get("cache_hit", False),
        retrieval_latency_ms=result.get("retrieval_latency_ms", 0),
        generation_latency_ms=result.get("generation_latency_ms", 0),
        total_latency_ms=result.get("total_latency_ms", 0),
        token_usage=TokenUsage(
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0),
        ),
        faithfulness_flag=flag,
        faithfulness_confidence=result.get("faithfulness_confidence", 0.0),
        context_chunks_used=result.get("context_chunks_used", 0),
        confidence_message=confidence_messages.get(flag),
    )


# ── GET /documents ─────────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all documents and their ingestion status.

    Status values:
      "processing" — ingestion in progress
      "ready"      — fully indexed, queryable
      "failed"     — ingestion failed (see error field)
    """
    registry = get_registry()
    documents = [
        DocumentInfo(
            doc_id=doc_id,
            filename=meta.get("filename", "unknown"),
            status=meta.get("status", "unknown"),
            pages=meta.get("pages"),
            small_chunks=meta.get("small_chunks"),
            large_chunks=meta.get("large_chunks"),
            ingested_at=meta.get("ingested_at"),
        )
        for doc_id, meta in registry.items()
    ]
    return DocumentListResponse(documents=documents, total=len(documents))


# ── GET /metrics ───────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Latency percentiles and cache statistics.

    Use this to track:
      - P95 total latency (target: <2000ms)
      - P95 retrieval latency (target: <200ms)
      - Cache hit rate (target: >30%)
    """
    tracker = get_tracker()
    cache = get_cache()

    latency_stats = tracker.stats
    cache_stats = cache.stats

    return MetricsResponse(
        total_queries=latency_stats["total_queries"],
        total_p50_ms=latency_stats["total_p50_ms"],
        total_p95_ms=latency_stats["total_p95_ms"],
        total_p99_ms=latency_stats["total_p99_ms"],
        retrieval_p50_ms=latency_stats["retrieval_p50_ms"],
        retrieval_p95_ms=latency_stats["retrieval_p95_ms"],
        retrieval_p99_ms=latency_stats["retrieval_p99_ms"],
        meets_p95_target=latency_stats["meets_p95_target"],
        meets_retrieval_target=latency_stats["meets_retrieval_target"],
        cache_hit_rate=cache_stats["hit_rate"],
        cache_entries=cache_stats["entries_stored"],
        cache_total_queries=cache_stats["total_queries"],
        avg_tokens_per_query=latency_stats["avg_tokens_per_query"],
    )
