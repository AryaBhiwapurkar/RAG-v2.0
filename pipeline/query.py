"""
pipeline/query.py — Query pipeline: question → retrieved context → answer.

FLOW (per user question):
  User question
    ↓ cache.py: semantic lookup (cosine > 0.92 → return cached result)
    ↓ embedder.py: embed query → 768-dim vector
    ↓ load indexes for requested doc_ids
    ↓ retriever.py: BM25 + FAISS + RRF + rerank → top-4 large chunks
    ↓ generator.py: structured prompt + Gemini 1.5 Flash → answer
    ↓ faithfulness post-check
    ↓ cache.py: store result
    ↓ latency_tracker.py: record timing
    ↓ Return: answer + sources + latency + faithfulness flag

MULTI-DOC SUPPORT:
  If doc_ids is a list of multiple doc IDs, we retrieve from each
  document's index independently, then pool all retrieved chunks
  and pass the combined context to the LLM.
  This is simple and effective at our scale (10-20 docs).
  At V3 scale (1M docs), you'd need filtered vector DB search instead.
"""

import time
import logging
from config.settings import settings
from core.embedder import embed_text
from core.retriever import hybrid_retrieve
from core.generator import generate_answer
from storage.vector_store import load_index, index_exists
from storage.cache import get_cache
from evaluation.latency_tracker import get_tracker

logger = logging.getLogger(__name__)


def run_query(
    question: str,
    doc_ids: list[str] | None = None,
) -> dict:
    """
    Full query pipeline from raw question to structured response.

    Args:
        question: User's question string.
        doc_ids: List of document IDs to search. If None, searches all
                 available documents (from the registry).

    Returns:
        Dict with:
          - answer (str): Generated answer
          - sources (list[str]): Source doc_ids
          - cache_hit (bool): Whether result came from cache
          - retrieval_latency_ms (float)
          - generation_latency_ms (float)
          - total_latency_ms (float)
          - token_usage (dict)
          - faithfulness_flag (str): "high" | "medium" | "low"
          - faithfulness_confidence (float)
          - context_chunks_used (int): How many chunks were sent to LLM
    """
    t_total_start = time.time()
    cache = get_cache()
    tracker = get_tracker()

    # ── Resolve doc_ids ────────────────────────────────────────────────────
    if doc_ids is None or len(doc_ids) == 0:
        # Search all available documents
        from pipeline.ingest import get_registry
        registry = get_registry()
        doc_ids = [
            doc_id for doc_id, meta in registry.items()
            if meta.get("status") == "ready"
        ]

    if not doc_ids:
        return {
            "answer": "No documents have been ingested yet. Please upload a PDF first.",
            "sources": [],
            "cache_hit": False,
            "retrieval_latency_ms": 0,
            "generation_latency_ms": 0,
            "total_latency_ms": 0,
            "token_usage": {},
            "faithfulness_flag": "low",
            "faithfulness_confidence": 0.0,
            "context_chunks_used": 0,
        }

    # ── Embed query ─────────────────────────────────────────────────────────
    query_vector = embed_text(question)

    # ── Cache lookup ────────────────────────────────────────────────────────
    cached = cache.get(query_vector)
    if cached:
        total_ms = (time.time() - t_total_start) * 1000
        cached["total_latency_ms"] = round(total_ms, 1)
        tracker.record(total_ms, cached.get("retrieval_latency_ms", 0))
        return cached

    # ── Retrieval (one index per doc, pool results) ─────────────────────────
    t_retrieval_start = time.time()
    all_context_chunks = []

    for doc_id in doc_ids:
        if not index_exists(doc_id):
            logger.warning(f"Index not found for doc_id='{doc_id}', skipping")
            continue

        try:
            faiss_index, bm25_index, small_chunks, large_chunks = load_index(doc_id)
            doc_chunks = hybrid_retrieve(
                query=question,
                query_vector=query_vector,
                faiss_index=faiss_index,
                bm25_index=bm25_index,
                small_chunks=small_chunks,
                large_chunks=large_chunks,
            )
            all_context_chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"Retrieval failed for doc_id='{doc_id}': {e}")
            continue

    retrieval_ms = (time.time() - t_retrieval_start) * 1000
    logger.info(
        f"Retrieval complete: {len(all_context_chunks)} context chunks "
        f"from {len(doc_ids)} docs in {retrieval_ms:.0f}ms"
    )

    # If multi-doc, limit total context to rerank_top_k chunks to avoid
    # overwhelming the LLM with too much context
    if len(all_context_chunks) > settings.rerank_top_k:
        all_context_chunks = all_context_chunks[:settings.rerank_top_k]

    # ── Generation ──────────────────────────────────────────────────────────
    gen_result = generate_answer(question, all_context_chunks)

    total_ms = (time.time() - t_total_start) * 1000
    tracker.record(total_ms, retrieval_ms)

    result = {
        "answer": gen_result["answer"],
        "sources": gen_result["sources"],
        "cache_hit": False,
        "retrieval_latency_ms": round(retrieval_ms, 1),
        "generation_latency_ms": gen_result["generation_latency_ms"],
        "total_latency_ms": round(total_ms, 1),
        "token_usage": gen_result["token_usage"],
        "faithfulness_flag": gen_result["faithfulness_flag"],
        "faithfulness_confidence": gen_result["faithfulness_confidence"],
        "context_chunks_used": len(all_context_chunks),
    }

    # ── Cache store ─────────────────────────────────────────────────────────
    cache.set(question, query_vector, result)

    logger.info(
        f"Query complete in {total_ms:.0f}ms | "
        f"faithfulness={gen_result['faithfulness_flag']} | "
        f"tokens={gen_result['token_usage'].get('total_tokens', '?')}"
    )
    return result
