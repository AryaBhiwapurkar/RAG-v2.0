"""
core/retriever.py — Hybrid retrieval: BM25 + FAISS + RRF + cross-encoder rerank.

PIPELINE:
  Query
    ↓
  [BM25 sparse search]    → top-20 by keyword score
  [FAISS dense search]    → top-20 by semantic similarity
    ↓
  [RRF fusion]            → combine rankings → top-20 deduplicated
    ↓
  [Cross-encoder rerank]  → re-score top-20 jointly → top-4
    ↓
  [Parent lookup]         → swap small chunks → large parent chunks
    ↓
  Final 4 large chunks → sent to LLM

WHY EACH STAGE:
  BM25   — catches exact term matches (codes, names, numbers, abbreviations)
  FAISS  — catches semantic similarity (synonyms, paraphrases, intent)
  RRF    — rank-based fusion, no score normalisation needed
  Rerank — cross-encoder sees (query, chunk) jointly → most accurate scoring
  Parent — small chunks retrieved, large chunks generated (context quality)

RRF FORMULA:
  score(chunk) = Σ  1 / (k + rank_i)
  where k=60 (prevents top rank from dominating), rank_i = position in list i
"""

import numpy as np
import logging
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from config.settings import settings
from core.preprocessor import preprocess_query

logger = logging.getLogger(__name__)

# Load cross-encoder once at module import (heavy model, don't reload per query)
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading cross-encoder: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


# ── INDIVIDUAL RETRIEVERS ──────────────────────────────────────────────────────

def dense_search(
    query_vector: np.ndarray,
    faiss_index: faiss.Index,
    small_chunks: list[dict],
    top_k: int,
) -> list[tuple[int, float]]:
    """
    FAISS semantic similarity search.

    Args:
        query_vector: 768-dim query embedding (will be normalised).
        faiss_index: Loaded FAISS index for this document.
        small_chunks: List of chunk dicts (same order as FAISS index).
        top_k: Number of results to return.

    Returns:
        List of (chunk_index, distance) sorted by distance ascending (lower=better).
    """
    # Normalise query vector (vectors in index are also normalised)
    q = query_vector.copy().reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q)

    actual_k = min(top_k, faiss_index.ntotal)
    distances, indices = faiss_index.search(q, actual_k)

    results = [
        (int(idx), float(dist))
        for idx, dist in zip(indices[0], distances[0])
        if idx >= 0  # FAISS returns -1 for empty slots
    ]
    logger.debug(f"FAISS returned {len(results)} candidates")
    return results


def sparse_search(
    query: str,
    bm25_index: BM25Okapi,
    small_chunks: list[dict],
    top_k: int,
) -> list[tuple[int, float]]:
    """
    BM25 keyword search.

    Args:
        query: Raw query string (will be preprocessed internally).
        bm25_index: Fitted BM25Okapi instance.
        small_chunks: List of chunk dicts (same order as BM25 corpus).
        top_k: Number of results to return.

    Returns:
        List of (chunk_index, bm25_score) sorted by score descending.
    """
    preprocessed = preprocess_query(query)
    tokens = preprocessed.lower().split()

    scores = bm25_index.get_scores(tokens)  # Shape: (n_chunks,)

    # Get top-k indices sorted by score descending
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    logger.debug(f"BM25 returned {len(results)} candidates with score > 0")
    return results


# ── RRF FUSION ────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = None,
) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Formula: score(chunk) = Σ 1 / (k + rank_i)
    - k=60 prevents top-ranked items from dominating
    - rank is 1-indexed (first result = rank 1)
    - Works on chunk indices (not text) so deduplication is exact

    Args:
        ranked_lists: List of ranked result lists, each [(chunk_idx, score), ...]
        k: RRF constant (defaults to settings value = 60)

    Returns:
        List of (chunk_idx, rrf_score) sorted by rrf_score descending.
    """
    if k is None:
        k = settings.rrf_k_constant

    rrf_scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk_idx, _) in enumerate(ranked_list, start=1):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (k + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"RRF fused {len(fused)} unique candidates")
    return fused


# ── CROSS-ENCODER RERANKER ────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: list[tuple[int, float]],
    small_chunks: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Cross-encoder reranking of RRF candidates.

    WHY cross-encoder beats bi-encoder here:
      Bi-encoder (FAISS): encodes query and chunk SEPARATELY → fast but less accurate
      Cross-encoder: takes (query, chunk) as a PAIR → sees interaction → more accurate
      Too slow for all chunks; perfect for top-20 candidates

    Args:
        query: Original user query (not preprocessed — reranker needs full context).
        candidates: RRF results [(chunk_idx, rrf_score), ...].
        small_chunks: Full list of small chunk dicts.
        top_k: Final number of chunks to return.

    Returns:
        List of top-k small chunk dicts, sorted by cross-encoder score descending.
    """
    reranker = _get_reranker()

    # Build (query, chunk_text) pairs for cross-encoder
    pairs = []
    valid_chunks = []
    for chunk_idx, _ in candidates:
        if chunk_idx < len(small_chunks):
            pairs.append([query, small_chunks[chunk_idx]["text"]])
            valid_chunks.append(small_chunks[chunk_idx])

    if not pairs:
        return []

    scores = reranker.predict(pairs)  # Returns numpy array of relevance scores

    # Combine with chunks and sort by score
    scored = sorted(zip(scores, valid_chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]

    logger.info(f"Reranker selected top-{len(top_chunks)} from {len(pairs)} candidates")
    return top_chunks


# ── PARENT LOOKUP ─────────────────────────────────────────────────────────────

def get_parent_chunks(
    small_chunks: list[dict],
    large_chunks: dict[str, dict],
) -> list[dict]:
    """
    Swap small retrieved chunks for their large parent chunks.

    This is the key parent-child trick:
    - Retrieval found small chunks (precise match)
    - We now look up each small chunk's parent_id
    - Return the large parent chunks (rich context for LLM)
    - Deduplicate: two small chunks from the same parent → return parent once

    Args:
        small_chunks: Top-k small chunks from reranker.
        large_chunks: Full dict of large chunks keyed by chunk_id.

    Returns:
        Deduplicated list of large parent chunks.
    """
    seen_parent_ids = set()
    parents = []

    for small in small_chunks:
        parent_id = small.get("parent_id")
        if parent_id and parent_id not in seen_parent_ids:
            parent = large_chunks.get(parent_id)
            if parent:
                parents.append(parent)
                seen_parent_ids.add(parent_id)

    logger.debug(f"Parent lookup: {len(small_chunks)} small → {len(parents)} unique parents")
    return parents


# ── FULL HYBRID RETRIEVE ──────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    query_vector: np.ndarray,
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    small_chunks: list[dict],
    large_chunks: dict[str, dict],
) -> list[dict]:
    """
    Full hybrid retrieval pipeline for a single document index.

    Runs: BM25 + FAISS → RRF → rerank → parent lookup

    Args:
        query: Original user query string.
        query_vector: Pre-computed 768-dim query embedding.
        faiss_index: Document's FAISS index.
        bm25_index: Document's BM25 index.
        small_chunks: Document's small chunk list.
        large_chunks: Document's large chunk dict.

    Returns:
        List of large parent chunks (final context for LLM).
    """
    # 1. Dense retrieval
    dense_results = dense_search(
        query_vector, faiss_index, small_chunks, settings.faiss_top_k
    )

    # 2. Sparse retrieval
    sparse_results = sparse_search(
        query, bm25_index, small_chunks, settings.bm25_top_k
    )

    # 3. RRF fusion
    fused = reciprocal_rank_fusion([dense_results, sparse_results])
    top_candidates = fused[:20]  # Take top-20 into reranker

    # 4. Cross-encoder rerank → top-4
    top_small = rerank(query, top_candidates, small_chunks, settings.rerank_top_k)

    # 5. Parent chunk lookup
    context_chunks = get_parent_chunks(top_small, large_chunks)

    return context_chunks
