"""
storage/vector_store.py — Per-document FAISS index management.

WHY per-document indexes (not one shared index):
  - Can delete one document without rebuilding everything
  - Can query a single doc or all docs (multi-doc support)
  - Each index is small and fast

WHAT'S STORED per document (data/indexes/{doc_id}/):
  faiss.index      — FAISS IndexFlatL2 (exact L2 distance search)
  bm25.pkl         — Serialised BM25Okapi index
  small_chunks.pkl — List of small chunk dicts (text + metadata)
  large_chunks.pkl — Dict of large chunk dicts (keyed by chunk_id)

WHY IndexFlatL2 not HNSW:
  - IndexFlatL2 is exact search (no approximation error)
  - Fast enough for <100k vectors (our scale: 10-20 docs × ~500 chunks = ~10k)
  - HNSW is for V3 at 500k+ vectors (as documented in design doc)

DISTANCE vs SIMILARITY:
  FAISS IndexFlatL2 returns L2 (Euclidean) distances, NOT cosine similarity.
  For normalised vectors these are equivalent (lower L2 = higher cosine).
  We normalise vectors at search time for consistent ranking.
"""

import faiss
import pickle
import numpy as np
import logging
from pathlib import Path
from rank_bm25 import BM25Okapi
from config.settings import settings

logger = logging.getLogger(__name__)


def _doc_dir(doc_id: str) -> Path:
    """Return and create the directory for a document's indexes."""
    d = settings.indexes_dir / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_index(
    doc_id: str,
    vectors: np.ndarray,
    small_chunks: list[dict],
    large_chunks: dict[str, dict],
    bm25_index: BM25Okapi,
) -> None:
    """
    Persist all index components for a document to disk.

    Args:
        doc_id: Unique document identifier.
        vectors: 2D float32 array shape (N, 768) — one vector per small chunk.
        small_chunks: List of small chunk dicts (same order as vectors).
        large_chunks: Dict of large chunk dicts keyed by chunk_id.
        bm25_index: Fitted BM25Okapi instance.
    """
    d = _doc_dir(doc_id)

    # Build and save FAISS index
    dim = vectors.shape[1]                        # Should be 768
    index = faiss.IndexFlatL2(dim)
    # Normalise vectors before adding (makes L2 ≈ cosine similarity ranking)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    faiss.write_index(index, str(d / "faiss.index"))

    # Save BM25, chunks
    with open(d / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    with open(d / "small_chunks.pkl", "wb") as f:
        pickle.dump(small_chunks, f)
    with open(d / "large_chunks.pkl", "wb") as f:
        pickle.dump(large_chunks, f)

    logger.info(f"Saved index for doc '{doc_id}': {index.ntotal} vectors in {d}")


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_index(doc_id: str) -> tuple[faiss.Index, BM25Okapi, list[dict], dict]:
    """
    Load all index components for a document from disk.

    Returns:
        Tuple of (faiss_index, bm25_index, small_chunks, large_chunks)

    Raises:
        FileNotFoundError if doc_id hasn't been ingested yet.
    """
    d = _doc_dir(doc_id)

    index_path = d / "faiss.index"
    if not index_path.exists():
        raise FileNotFoundError(f"No index found for doc_id='{doc_id}'. Ingest it first.")

    faiss_index = faiss.read_index(str(index_path))

    with open(d / "bm25.pkl", "rb") as f:
        bm25_index = pickle.load(f)
    with open(d / "small_chunks.pkl", "rb") as f:
        small_chunks = pickle.load(f)
    with open(d / "large_chunks.pkl", "rb") as f:
        large_chunks = pickle.load(f)

    logger.info(f"Loaded index for doc '{doc_id}': {faiss_index.ntotal} vectors")
    return faiss_index, bm25_index, small_chunks, large_chunks


# ── DELETE ─────────────────────────────────────────────────────────────────────

def delete_index(doc_id: str) -> None:
    """
    Remove all stored index files for a document.
    Used when a user deletes a document from the system.
    """
    import shutil
    d = _doc_dir(doc_id)
    if d.exists():
        shutil.rmtree(d)
        logger.info(f"Deleted index for doc '{doc_id}'")
    else:
        logger.warning(f"Tried to delete non-existent index for doc '{doc_id}'")


# ── EXISTS CHECK ───────────────────────────────────────────────────────────────

def index_exists(doc_id: str) -> bool:
    """Return True if a FAISS index exists for this doc_id."""
    return (settings.indexes_dir / doc_id / "faiss.index").exists()


def build_bm25_index(small_chunks: list[dict]) -> BM25Okapi:
    """
    Build a BM25 index from a list of small chunks.

    Tokenises each chunk's text by whitespace (simple but effective for BM25).
    BM25Okapi is the standard variant: handles term frequency saturation
    and document length normalisation.

    Args:
        small_chunks: List of chunk dicts with 'text' key.

    Returns:
        Fitted BM25Okapi instance ready for .get_scores() calls.
    """
    tokenised_corpus = [chunk["text"].lower().split() for chunk in small_chunks]
    return BM25Okapi(tokenised_corpus)
