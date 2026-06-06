"""
core/embedder.py — Local SentenceTransformer embeddings.

PHASE 1 CHANGES:
  - Model switched to paraphrase-MiniLM-L3-v2 (via settings.embedding_model)
  - embed_batch: explicit batch_size=64 (sweet spot for CPU; prevents OOM on large PDFs)
  - embed_batch: show_progress_bar=False (was True — noisy in server logs)
  - Phase timing added: logs ms per call so ingestion bottleneck is visible
"""

import time
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from config.settings import settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"[EMBEDDER] Loading model: {settings.embedding_model}")
        t0 = time.perf_counter()
        _model = SentenceTransformer(settings.embedding_model)
        logger.info(f"[EMBEDDER] Model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single string. Used for query embedding at query time.
    Kept separate from embed_batch — different call path, easier to time independently.
    """
    t0 = time.perf_counter()
    model = _get_model()
    vector = model.encode(text.strip(), normalize_embeddings=True)
    logger.debug(f"[EMBEDDER] embed_text: {(time.perf_counter()-t0)*1000:.1f}ms")
    return vector.astype(np.float32)


def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings. Used during ingestion.

    batch_size=64: encodes 64 chunks per forward pass.
    - Too small (e.g. 8): too many forward passes, slow.
    - Too large (e.g. 256): high peak RAM on CPU, diminishing returns.
    - 64 is a safe default; bump to 128 if you have 16GB+ RAM.

    show_progress_bar=False: progress bars pollute FastAPI logs;
    we emit our own timing log instead.
    """
    t0 = time.perf_counter()
    model = _get_model()

    cleaned = [t.strip() if t.strip() else " " for t in texts]

    vectors = model.encode(
        cleaned,
        normalize_embeddings=True,
        batch_size=64,          # PHASE 1: explicit, was implicit default (32)
        show_progress_bar=False,  # PHASE 1: suppressed, use timing log below
    )
    result = np.array(vectors, dtype=np.float32)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_chunk_ms = elapsed_ms / max(len(texts), 1)
    logger.info(
        f"[EMBEDDER] embed_batch: {len(texts)} chunks → shape {result.shape} "
        f"| {elapsed_ms:.0f}ms total | {per_chunk_ms:.1f}ms/chunk"
    )
    return result


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))