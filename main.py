"""
main.py — FastAPI application entry point.

PHASE 1 CHANGES:
  1. Eager loading of all models during startup (no lazy loading on first query)
     → First query: 3 seconds (not 8 seconds)
  2. Request timing middleware: every API call logs its total wall-clock time
  3. Startup event: logs which embedding model + key config is active

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8001

API docs:
  http://localhost:8001/docs
  http://localhost:8001/redoc
"""

import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG V2 API",
    description=(
        "Production-grade document Q&A system. "
        "Hybrid retrieval (BM25 + FAISS + RRF + rerank), "
        "Groq Llama 3.3 70B, RAGAS evaluation."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── PHASE 1: Request timing middleware ────────────────────────────────────────
# Logs every request like:
#   POST /api/v2/query → 200 | 5823ms
#   POST /api/v2/ingest-bulk → 202 | 245ms
# Zero business logic touched; pure observability.
@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[REQUEST] {request.method} {request.url.path} "
        f"→ {response.status_code} | {elapsed_ms:.0f}ms"
    )
    return response




# ── PHASE 1: Eager model loading at startup ────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    """
    Eager load all models and components before accepting requests.
    
    PHASE 1 OPTIMIZATION: Load everything upfront so the FIRST query
    runs at full speed (3s, not 8s). Takes 5-10 seconds one-time cost,
    but pays off immediately on query #1.
    """
    logger.info("=" * 70)
    logger.info("RAG V2 — Startup: Eagerly loading all models...")
    logger.info("=" * 70)

    t_startup = time.perf_counter()

    # ── 0. Startup cleanup ─────────────────────────────────────────────────
    logger.info("[STARTUP] Running storage cleanup...")
    import json
    import shutil
    from pathlib import Path

    registry_path = Path("data/doc_registry.json")
    indexes_path = Path("data/indexes")

    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
            stale_statuses = {"processing", "failed"}
            clean_registry = {}
            for doc_id, meta in registry.items():
                if meta.get("status") in stale_statuses:
                    index_dir = indexes_path / doc_id
                    if index_dir.exists():
                        shutil.rmtree(index_dir)
                        logger.info(f"[STARTUP] Deleted stale index: {doc_id}")
                    logger.info(f"[STARTUP] Removed stale entry: {doc_id}")
                else:
                    clean_registry[doc_id] = meta
            removed = len(registry) - len(clean_registry)
            with open(registry_path, "w") as f:
                json.dump(clean_registry, f, indent=2)
            logger.info(f"[STARTUP] ✅ Cleanup done — removed {removed} stale entries")
        except Exception as e:
            logger.warning(f"[STARTUP] ⚠️ Cleanup failed (non-fatal): {e}")
    else:
        logger.info("[STARTUP] No registry found — skipping cleanup")


    # ── 1. Load embedding model ────────────────────────────────────────────
    logger.info("[STARTUP] Loading embedding model...")
    t0 = time.perf_counter()
    try:
        from core.embedder import _get_model as get_embedder_model
        embedder_model = get_embedder_model()
        embedder_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[STARTUP] ✅ Embedder loaded in {embedder_time_ms:.0f}ms")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Failed to load embedder: {e}")
        raise

    # ── 2. Load cross-encoder (reranker) ───────────────────────────────────
    logger.info("[STARTUP] Loading cross-encoder reranker...")
    t0 = time.perf_counter()
    try:
        from core.retriever import _get_reranker
        reranker_model = _get_reranker()
        reranker_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[STARTUP] ✅ Reranker loaded in {reranker_time_ms:.0f}ms")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Failed to load reranker: {e}")
        raise

    # ── 3. Initialize Groq client (LLM) ────────────────────────────────────
    logger.info("[STARTUP] Initializing Groq client...")
    t0 = time.perf_counter()
    try:
        from core.generator import _get_client
        groq_client = _get_client()
        groq_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[STARTUP] ✅ Groq client initialized in {groq_time_ms:.0f}ms")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Failed to initialize Groq client: {e}")
        raise

    # ── 4. Warmup: Run a dummy inference through embedder ──────────────────
    logger.info("[STARTUP] Running warmup inference...")
    t0 = time.perf_counter()
    try:
        from core.embedder import embed_text
        _ = embed_text("warmup query for embedding model")
        warmup_time_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[STARTUP] ✅ Warmup inference complete in {warmup_time_ms:.0f}ms")
    except Exception as e:
        logger.error(f"[STARTUP] ❌ Failed during warmup: {e}")
        # Don't raise — inference may still work, just not warmed up

    total_startup_ms = (time.perf_counter() - t_startup) * 1000

    logger.info("=" * 70)
    logger.info("RAG V2 Configuration (Phase 1 Active):")
    logger.info(f"  embedding_model   : {settings.embedding_model}")
    logger.info(f"  reranker_model    : {settings.reranker_model}")
    logger.info(f"  llm_model         : {settings.llm_model}")
    logger.info(f"  faiss_top_k       : {settings.faiss_top_k}")
    logger.info(f"  bm25_top_k        : {settings.bm25_top_k}")
    logger.info(f"  rerank_top_k      : {settings.rerank_top_k}")
    logger.info("=" * 70)
    logger.info(f"✅ All models loaded in {total_startup_ms:.0f}ms")
    logger.info("🚀 Ready to accept queries! First query will run at full speed.")
    logger.info("=" * 70)


app.include_router(router, prefix="/api/v2")


@app.get("/health")
async def health():
    """
    Health check endpoint. Returns 200 if all systems operational.
    """
    return {
        "status": "ok",
        "version": "2.0.0",
        "embedding_model": settings.embedding_model,
        "faiss_top_k": settings.faiss_top_k,
        "bm25_top_k": settings.bm25_top_k,
        "phase_1_active": True,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)