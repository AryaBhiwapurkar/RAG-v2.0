"""
storage/cache.py — Semantic in-memory query cache.

WHY SEMANTIC (not exact string match):
  "What is the punishment for murder?" and "Murder punishment India?"
  are the same question phrased differently. Exact cache misses both.
  Semantic cache embeds both → cosine similarity >0.92 → cache hit.

WHY IN-MEMORY (not Redis):
  At our scale (10-20 docs, <50 concurrent users) Redis is overkill.
  In-memory is zero infra, zero config, instant.
  Trade-off: cache resets on server restart (documented in design doc).

HOW IT WORKS:
  - Each cache entry stores: (query_vector, query_text, result)
  - On new query: embed it → compare cosine against all stored vectors
  - If any similarity > threshold (0.92) → return cached result
  - Otherwise → run full pipeline → store result

EXPECTED IMPACT: ~40% reduction in LLM API calls (per design doc).
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from core.embedder import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query_text: str
    query_vector: np.ndarray
    result: dict           # Full pipeline result dict
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    """
    Thread-safe-ish semantic query cache (single-process).
    For multi-process deployments, replace with Redis (V3).
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self._entries: list[CacheEntry] = []
        self._total_queries = 0
        self._cache_hits = 0

    def get(self, query_vector: np.ndarray) -> dict | None:
        """
        Check if a semantically similar query is cached.

        Args:
            query_vector: 768-dim embedding of the incoming query.

        Returns:
            Cached result dict if hit, None if miss.
        """
        self._total_queries += 1

        for entry in self._entries:
            sim = cosine_similarity(query_vector, entry.query_vector)
            if sim >= self.threshold:
                entry.hit_count += 1
                self._cache_hits += 1
                logger.info(
                    f"Cache HIT (similarity={sim:.4f}) for query "
                    f"'{entry.query_text[:60]}...'"
                )
                # Return a copy with cache metadata added
                result = dict(entry.result)
                result["cache_hit"] = True
                result["cache_similarity"] = round(sim, 4)
                return result

        logger.debug("Cache MISS — running full pipeline")
        return None

    def set(self, query_text: str, query_vector: np.ndarray, result: dict) -> None:
        """
        Store a query result in the cache.

        Args:
            query_text: Original query string (for logging).
            query_vector: 768-dim embedding of the query.
            result: Full pipeline result dict to cache.
        """
        entry = CacheEntry(
            query_text=query_text,
            query_vector=query_vector.copy(),
            result=result,
        )
        self._entries.append(entry)
        logger.debug(f"Cached result for query '{query_text[:60]}...' "
                     f"(cache size: {len(self._entries)})")

    def clear(self) -> None:
        """Clear all cached entries (e.g. when documents change)."""
        self._entries.clear()
        logger.info("Cache cleared")

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a float 0.0–1.0."""
        if self._total_queries == 0:
            return 0.0
        return self._cache_hits / self._total_queries

    @property
    def stats(self) -> dict:
        return {
            "total_queries": self._total_queries,
            "cache_hits": self._cache_hits,
            "hit_rate": round(self.hit_rate, 3),
            "entries_stored": len(self._entries),
        }


# Module-level singleton — shared across the entire FastAPI app process
_cache = SemanticCache()


def get_cache() -> SemanticCache:
    """Return the global cache instance."""
    return _cache
