"""
pipeline/query.py — Query pipeline: question → retrieved context → answer.

FLOW (per user question):
  User question
    ↓ cache.py: semantic lookup (cosine > 0.92 → return cached result)
    ↓ embedder.py: embed query → 768-dim vector
    ↓ load indexes for requested doc_ids
    ↓ retriever.py: BM25 + FAISS + RRF + rerank → top-k small chunks
    ↓ [multi-doc only] global cross-encoder rerank on small chunks → top-4
    ↓ parent lookup → swap small chunks → large parent chunks (300→1200 chars)
    ↓ generator.py: structured prompt + Groq LLM → answer
    ↓ faithfulness post-check
    ↓ cache.py: store result
    ↓ latency_tracker.py: record timing
    ↓ Return: answer + sources + latency + faithfulness flag

MULTI-DOC BALANCE FIX (Phase 1):
  OLD: each doc returned top-4 large chunks → big docs drowned small docs.
       Final limit was a dumb slice with no reranking.
  NEW: each doc returns top-2 small chunks (_PER_DOC_TOP_K).
       Global cross-encoder rerank on small chunks → final top-4 small chunks.
       Parent lookup done ONCE after global rerank.
       Small docs guaranteed representation; best chunks win globally.

PHASE 2 FIX (cross-encoder input size):
  OLD: global rerank scored (query, parent_chunk) — 1200 chars, out of
       distribution for ms-marco which was trained on ~300 char passages.
  NEW: global rerank scores (query, small_chunk) — 300 chars, correct usage.
       Parent lookup moved to after global rerank so context richness is
       still preserved for the LLM.
"""

import time
import logging
from config.settings import settings
from core.embedder import embed_text
from core.retriever import hybrid_retrieve, rerank as cross_rerank, get_parent_chunks
from core.generator import generate_answer
from storage.vector_store import load_index, index_exists
from storage.cache import get_cache
from evaluation.latency_tracker import get_tracker

logger = logging.getLogger(__name__)

# Each doc contributes this many small chunks before the global rerank.
# Keeps small docs from being drowned by large ones.
# Tune if you add more documents — e.g. 10 docs → consider top-1 per doc.
_PER_DOC_TOP_K = 2


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

    # ── Retrieval ───────────────────────────────────────────────────────────
    t_retrieval_start = time.time()
    is_multi_doc = len(doc_ids) > 1

    # We need large_chunks later for parent lookup — store per doc_id
    all_small_chunks: list[dict] = []
    large_chunks_registry: dict[str, dict] = {}  # parent_id → large chunk, across all docs

    for doc_id in doc_ids:
        if not index_exists(doc_id):
            logger.warning(f"Index not found for doc_id='{doc_id}', skipping")
            continue

        try:
            faiss_index, bm25_index, small_chunks, large_chunks = load_index(doc_id)

            if is_multi_doc:
                # Return small chunks so global rerank scores them correctly
                # (300 chars — what ms-marco was trained on, not 1200 char parents)
                doc_small = hybrid_retrieve(
                    query=question,
                    query_vector=query_vector,
                    faiss_index=faiss_index,
                    bm25_index=bm25_index,
                    small_chunks=small_chunks,
                    large_chunks=large_chunks,
                    override_rerank_top_k=_PER_DOC_TOP_K,
                    return_small_chunks=True,   # ← skip parent lookup here
                )
                all_small_chunks.extend(doc_small)
                large_chunks_registry.update(large_chunks)  # accumulate for later
            else:
                # Single doc: normal path — parent lookup inside hybrid_retrieve
                doc_chunks = hybrid_retrieve(
                    query=question,
                    query_vector=query_vector,
                    faiss_index=faiss_index,
                    bm25_index=bm25_index,
                    small_chunks=small_chunks,
                    large_chunks=large_chunks,
                )
                all_small_chunks = doc_chunks  # already large chunks for single doc

        except Exception as e:
            logger.error(f"Retrieval failed for doc_id='{doc_id}': {e}")
            continue

    retrieval_ms = (time.time() - t_retrieval_start) * 1000
    logger.info(
        f"Retrieval complete: {len(all_small_chunks)} candidates "
        f"from {len(doc_ids)} docs in {retrieval_ms:.0f}ms"
    )

    # ── Global rerank + parent lookup (multi-doc only) ──────────────────────
    if is_multi_doc and len(all_small_chunks) > 0:
        # Global cross-encoder rerank on small chunks (300 chars) — correct
        # input size for ms-marco. Picks best top-4 across all docs.
        candidates = [(i, 0.0) for i in range(len(all_small_chunks))]
        logger.debug(f"Accumulated large_chunks_registry: {len(large_chunks_registry)} parents")
        logger.debug(f"All small chunks before global rerank: {len(all_small_chunks)}")
        top_small = cross_rerank(
            question,
            candidates,
            all_small_chunks,
            settings.rerank_top_k,
        )
        logger.info(
            f"Global rerank: {len(all_small_chunks)} candidates → "
            f"top-{len(top_small)} after cross-encoder"
        )
        logger.debug(f"Top small after global rerank: {len(top_small)}")
        # Now do parent lookup once on the globally-ranked small chunks
        all_context_chunks = get_parent_chunks(top_small, large_chunks_registry)
        # Fallback: if parent lookup failed to return any parents (edge cases
        # where large_chunks_registry may be missing keys), use the reranked
        # small chunks as context so the LLM still receives useful text.
        if not all_context_chunks:
            logger.warning(
                "Multi-doc parent lookup returned 0 parents — falling back to using small chunks as context"
            )
            all_context_chunks = top_small
    else:
        # Single doc: all_small_chunks is already large chunks from hybrid_retrieve
        all_context_chunks = all_small_chunks

    # Final safety cap (shouldn't trigger normally but guards edge cases)
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