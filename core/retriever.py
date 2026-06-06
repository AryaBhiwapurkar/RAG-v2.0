"""
core/retriever.py — Hybrid retrieval: BM25 + FAISS + RRF + cross-encoder rerank.

PIPELINE:
  Query
    ↓
  [BM25 sparse search]    → top-10 by keyword score        ← PHASE 1: was 20
  [FAISS dense search]    → top-10 by semantic similarity  ← PHASE 1: was 20
    ↓ (parallel via ThreadPoolExecutor)                    ← PHASE 1: new
  [RRF fusion]            → combine rankings → top-20 deduplicated
    ↓
  [Cross-encoder rerank]  → re-score top-20 jointly → top-k
    ↓
  [Parent lookup]         → swap small chunks → large parent chunks
    ↓
  Final chunks → sent to LLM

PHASE 1 CHANGES:
  - BM25 + FAISS run in parallel (ThreadPoolExecutor) instead of sequentially.
  - Phase-wise timing on every stage.
  - override_rerank_top_k param wired up (was in signature but ignored before).

PHASE 2 FIX:
  - return_small_chunks param added to hybrid_retrieve().
    When True, skips parent lookup and returns small chunks instead.
    Used by query.py in multi-doc mode so the global cross-encoder rerank
    scores (query, small_chunk) pairs — what the model was trained on (300
    chars) — rather than (query, parent_chunk) pairs (1200 chars, out of
    distribution). Parent lookup happens once at the end in query.py after
    global rerank, so context richness is preserved.

RRF FORMULA:
  score(chunk) = Σ  1 / (k + rank_i)
  where k=60 (prevents top rank from dominating), rank_i = position in list i
"""

import time
import numpy as np
import logging
import faiss
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config.settings import settings
from core.preprocessor import preprocess_query

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None

_search_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retriever")


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"[RETRIEVER] Loading cross-encoder: {settings.reranker_model}")
        t0 = time.perf_counter()
        _reranker = CrossEncoder(settings.reranker_model)
        logger.info(f"[RETRIEVER] Cross-encoder loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
    return _reranker


# ── INDIVIDUAL RETRIEVERS ──────────────────────────────────────────────────────

def dense_search(
    query_vector: np.ndarray,
    faiss_index: faiss.Index,
    small_chunks: list[dict],
    top_k: int,
) -> list[tuple[int, float]]:
    """FAISS semantic similarity search."""
    q = query_vector.copy().reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q)

    actual_k = min(top_k, faiss_index.ntotal)
    distances, indices = faiss_index.search(q, actual_k)

    results = [
        (int(idx), float(dist))
        for idx, dist in zip(indices[0], distances[0])
        if idx >= 0
    ]
    logger.debug(f"[RETRIEVER] FAISS: {len(results)} candidates")
    return results


def sparse_search(
    query: str,
    bm25_index: BM25Okapi,
    small_chunks: list[dict],
    top_k: int,
) -> list[tuple[int, float]]:
    """BM25 keyword search."""
    preprocessed = preprocess_query(query)
    tokens = preprocessed.lower().split()

    scores = bm25_index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    logger.debug(f"[RETRIEVER] BM25: {len(results)} candidates with score > 0")
    return results


# ── RRF FUSION ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = None,
) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    Formula: score(chunk) = Σ 1 / (k + rank_i)
    """
    if k is None:
        k = settings.rrf_k_constant

    rrf_scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_idx, _) in enumerate(ranked_list, start=1):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"[RETRIEVER] RRF: {len(fused)} unique candidates after fusion")
    return fused


# ── CROSS-ENCODER RERANKER ────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[tuple[int, float]],
    small_chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """Cross-encoder reranking of RRF candidates."""
    reranker = _get_reranker()

    pairs = []
    valid_chunks = []
    for chunk_idx, _ in candidates:
        if chunk_idx < len(small_chunks):
            pairs.append([query, small_chunks[chunk_idx]["text"]])
            valid_chunks.append(small_chunks[chunk_idx])

    if not pairs:
        return []

    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, valid_chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]

    logger.debug(f"[RETRIEVER] Reranker: top-{len(top_chunks)} from {len(pairs)} candidates")
    return top_chunks


# ── PARENT LOOKUP ─────────────────────────────────────────────────────────────

def get_parent_chunks(
    small_chunks: list[dict],
    large_chunks: dict[str, dict],
) -> list[dict]:
    """Swap small retrieved chunks for their large parent chunks (deduplicated)."""
    seen_parent_ids = set()
    parents = []

    for small in small_chunks:
        parent_id = small.get("parent_id")
        if parent_id and parent_id not in seen_parent_ids:
            parent = large_chunks.get(parent_id)
            if parent:
                parents.append(parent)
                seen_parent_ids.add(parent_id)

    logger.debug(f"[RETRIEVER] Parent lookup: {len(small_chunks)} small → {len(parents)} parents")
    return parents


# ── FULL HYBRID RETRIEVE ──────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    query_vector: np.ndarray,
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    small_chunks: list[dict],
    large_chunks: dict[str, dict],
    override_rerank_top_k: int | None = None,
    return_small_chunks: bool = False,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline for a single document index.

    Args:
        override_rerank_top_k: If set, overrides settings.rerank_top_k for the
            per-doc rerank step. Used by query.py to cap per-doc contributions
            (e.g. top-2) before a global cross-encoder rerank across all docs.

        return_small_chunks: If True, skips the parent lookup step and returns
            small chunks (300 chars) instead of parent chunks (1200 chars).
            Used by query.py in multi-doc mode so the global cross-encoder rerank
            gets short passages — what ms-marco was trained on. Parent lookup
            is then done once in query.py after global rerank.
            If False (default), returns parent chunks as usual (single-doc path).

    Logs a timing breakdown at INFO level:
      [RETRIEVER] dense+sparse(parallel): 28ms | rrf: 1ms | rerank: 190ms | TOTAL: 219ms
    """
    timings: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    # ── 1 & 2. Dense + Sparse in PARALLEL ─────────────────────────────────
    t_search_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="search") as pool:
        future_dense = pool.submit(
            dense_search, query_vector, faiss_index, small_chunks, settings.faiss_top_k
        )
        future_sparse = pool.submit(
            sparse_search, query, bm25_index, small_chunks, settings.bm25_top_k
        )
        dense_results = future_dense.result()
        sparse_results = future_sparse.result()

    timings["dense+sparse(parallel)"] = (time.perf_counter() - t_search_start) * 1000

    # ── 3. RRF fusion ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    top_candidates = fused[:20]
    timings["rrf"] = (time.perf_counter() - t0) * 1000

    # ── 4. Cross-encoder rerank ────────────────────────────────────────────
    t0 = time.perf_counter()
    final_k = override_rerank_top_k if override_rerank_top_k is not None else settings.rerank_top_k
    top_small = rerank(query, top_candidates, small_chunks, final_k)
    timings["rerank"] = (time.perf_counter() - t0) * 1000

    # ── 5. Parent chunk lookup (skipped if return_small_chunks=True) ───────
    if return_small_chunks:
        # Caller (query.py multi-doc) will do global rerank on these small
        # chunks first, then call get_parent_chunks() itself after.
        timings["parent_lookup"] = 0.0
        timings["TOTAL"] = (time.perf_counter() - pipeline_start) * 1000
        timing_str = " | ".join(f"{k}: {v:.0f}ms" for k, v in timings.items())
        logger.info(f"[RETRIEVER] {timing_str}")
        return top_small

    t0 = time.perf_counter()
    context_chunks = get_parent_chunks(top_small, large_chunks)
    timings["parent_lookup"] = (time.perf_counter() - t0) * 1000

    timings["TOTAL"] = (time.perf_counter() - pipeline_start) * 1000
    timing_str = " | ".join(f"{k}: {v:.0f}ms" for k, v in timings.items())
    logger.info(f"[RETRIEVER] {timing_str}")

    return context_chunks