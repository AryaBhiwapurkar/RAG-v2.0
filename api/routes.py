"""
api/routes.py

PHASE 3 FIX — Async ingestion properly wired:
  - ingest_document() is sync (CPU-bound: PDF parsing, embedding, FAISS).
    Running it directly in BackgroundTasks blocks the event loop.
    Fix: asyncio.to_thread() pushes it to a thread pool, freeing the event loop.
  - Bulk ingestion uses asyncio.gather() — all files run concurrently in threads.
  - run_query() also pushed to thread (same reason — CPU-bound retrieval + LLM).

PHASE 3 FIX — Import fix:
  - get_registry() moved from pipeline.ingest to storage.registry (SQLite).
"""

import shutil
import logging
import time
import re
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from api.models import (
    QueryRequest, QueryResponse, IngestResponse, BulkIngestResponse,
    DocumentListResponse, DocumentInfo, MetricsResponse, TokenUsage
)
from pipeline.ingest import ingest_document
from storage.registry import get_registry
from pipeline.query import run_query
from storage.cache import get_cache
from evaluation.latency_tracker import get_tracker
from config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_QUERY_LENGTH = 512

_INJECTION_PATTERNS = re.compile(
    r"ignore (above|previous|all|prior) (instructions?|prompts?|context)|"
    r"jailbreak|DAN mode|you are now|disregard (your|all)|"
    r"forget (your|all) (instructions?|rules?)|"
    r"act as (an? )?(unrestricted|unfiltered|evil|dan)",
    re.IGNORECASE,
)


def _validate_query(question: str) -> None:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long ({len(question)} chars). Keep it under {MAX_QUERY_LENGTH}.",
        )
    if _INJECTION_PATTERNS.search(question):
        raise HTTPException(
            status_code=400,
            detail="Question contains disallowed patterns. Please ask a normal question.",
        )


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = settings.uploads_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # asyncio.to_thread: runs sync ingest_document in a thread pool.
    # Without this, BackgroundTasks runs it on the event loop → blocks all requests.
    background_tasks.add_task(asyncio.to_thread, ingest_document, save_path, file.filename)
    logger.info(f"Queued async ingestion for '{file.filename}'")

    return IngestResponse(
        doc_id="pending",
        filename=file.filename,
        status="processing",
        message=f"'{file.filename}' is being processed. Poll GET /documents to check when status='ready'.",
    )


@router.post("/ingest-bulk", response_model=BulkIngestResponse, status_code=202)
async def ingest_bulk(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDFs supported. Got: {file.filename}")

    save_paths = []
    for file in files:
        save_path = settings.uploads_dir / file.filename
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        save_paths.append((save_path, file.filename))

    background_tasks.add_task(_ingest_parallel, save_paths)
    logger.info(f"[BULK INGEST] Queued {len(files)} files")

    return BulkIngestResponse(
        files_count=len(files),
        status="processing",
        message=f"{len(files)} files queued. Poll GET /documents to check progress.",
    )


async def _ingest_parallel(save_paths: list[tuple]) -> None:
    """
    Fan out all ingestions concurrently.
    Each ingest_document() runs in its own thread via asyncio.to_thread().
    asyncio.gather() launches all threads simultaneously — no sequential waiting.
    """
    t_start = time.time()

    async def _one(save_path, filename):
        try:
            result = await asyncio.to_thread(ingest_document, save_path, filename)
            icon = "✅" if result.get("status") == "ready" else "❌"
            logger.info(f"[BULK] {icon} {filename}")
            return result
        except Exception as e:
            logger.error(f"[BULK] ❌ {filename}: {e}")
            return {"filename": filename, "status": "failed", "error": str(e)}

    results = await asyncio.gather(*[_one(sp, fn) for sp, fn in save_paths])
    elapsed = time.time() - t_start
    ok = sum(1 for r in results if r.get("status") == "ready")
    logger.info(f"[BULK] Done: {ok}/{len(results)} in {elapsed:.1f}s")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    _validate_query(request.question)

    # run_query is sync + CPU-bound — push to thread to free event loop
    result = await asyncio.to_thread(run_query, request.question, request.doc_ids)

    flag = result.get("faithfulness_flag", "high")
    confidence_messages = {
        "high": None,
        "medium": "Confidence: Medium — please verify against the source document.",
        "low": "Low confidence — I couldn't find a reliable answer. Check the document directly.",
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


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
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


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
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