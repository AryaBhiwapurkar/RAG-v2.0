"""
storage/cache.py — Semantic query cache backed by Redis.

WHY SEMANTIC (not exact string match):
  "What is the punishment for murder?" and "Murder punishment India?"
  are the same question phrased differently. Exact cache misses both.
  Semantic cache embeds both → cosine similarity > threshold → cache hit.

WHY REDIS (replacing in-memory dict):
  In-memory cache dies on every server restart — all warm cache is lost.
  Redis persists across restarts and is shared across workers if you ever
  run multiple uvicorn processes. Same semantic logic, better durability.
  TTL (default 24h) means stale results auto-expire without manual cleanup.

HOW IT WORKS:
  - Each cache entry stores: query_vector (as bytes) + result (as JSON)
  - Keys: "rag:cache:vectors" → Redis hash {entry_id: vector_bytes}
           "rag:cache:results" → Redis hash {entry_id: result_json}
           "rag:cache:meta"    → Redis hash {entry_id: query_text}
  - On new query: embed → compare cosine against all stored vectors
  - If any similarity > threshold (0.92) → return cached result
  - Otherwise → run full pipeline → store in Redis with TTL

FALLBACK:
  If Redis is not running, cache falls back to in-memory mode with a
  warning log. This means the server still works — you just lose
  persistence. No code changes needed in query.py either way.

EXPECTED IMPACT: ~40% reduction in LLM API calls (per design doc).
"""

import json
import uuid
import logging
import time
import numpy as np

from core.embedder import cosine_similarity
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Redis connection (lazy, with fallback) ─────────────────────────────────────

_redis_client = None
_redis_available = False


def _get_redis():
    """
    Lazy Redis connection. Returns client if available, None if not.
    Called once on first cache access — not at import time.
    """
    global _redis_client, _redis_available

    if _redis_client is not None:
        return _redis_client if _redis_available else None

    try:
        import redis
        client = redis.Redis(
            host=getattr(settings, "redis_host", "localhost"),
            port=getattr(settings, "redis_port", 6379),
            db=getattr(settings, "redis_db", 0),
            decode_responses=False,   # we store raw bytes for vectors
            socket_connect_timeout=2, # fail fast if Redis isn't running
        )
        client.ping()  # actually test the connection
        _redis_client = client
        _redis_available = True
        logger.info("[CACHE] Redis connected — persistent semantic cache active")
    except Exception as e:
        _redis_available = False
        logger.warning(
            f"[CACHE] Redis unavailable ({e}). "
            "Falling back to in-memory cache — cache will reset on restart."
        )

    return _redis_client if _redis_available else None


# Redis key constants
_KEY_VECTORS = "rag:cache:vectors"   # hash: entry_id → vector bytes
_KEY_RESULTS = "rag:cache:results"   # hash: entry_id → result JSON
_KEY_META    = "rag:cache:meta"      # hash: entry_id → query text
_KEY_HITS    = "rag:cache:hits"      # int: total cache hits
_KEY_QUERIES = "rag:cache:queries"   # int: total queries seen
_CACHE_TTL   = 60 * 60 * 24         # 24 hours in seconds


class SemanticCache:
    """
    Semantic query cache with Redis backend and in-memory fallback.

    Interface is identical to the old in-memory version — nothing outside
    this file needs to change.

    Redis storage layout:
      rag:cache:vectors  — hash of {entry_id: np.ndarray as bytes}
      rag:cache:results  — hash of {entry_id: json string}
      rag:cache:meta     — hash of {entry_id: query text}
      rag:cache:hits     — running hit counter
      rag:cache:queries  — running query counter
    """

    def __init__(self, similarity_threshold: float = None):
        self.threshold = similarity_threshold or getattr(
            settings, "cache_similarity_threshold", 0.92
        )
        # In-memory fallback storage (used when Redis is unavailable)
        self._fallback_entries: list[dict] = []
        self._fallback_hits = 0
        self._fallback_queries = 0

    # ── GET ────────────────────────────────────────────────────────────────

    def get(self, query_vector: np.ndarray) -> dict | None:
        """
        Check if a semantically similar query is cached.

        Args:
            query_vector: Embedding of the incoming query.

        Returns:
            Cached result dict if hit, None if miss.
        """
        r = _get_redis()

        if r is not None:
            return self._get_redis(r, query_vector)
        else:
            return self._get_fallback(query_vector)

    def _get_redis(self, r, query_vector: np.ndarray) -> dict | None:
        r.incr(_KEY_QUERIES)

        all_vectors = r.hgetall(_KEY_VECTORS)
        if not all_vectors:
            return None

        for entry_id_bytes, vec_bytes in all_vectors.items():
            entry_id = entry_id_bytes.decode()
            stored_vec = np.frombuffer(vec_bytes, dtype=np.float32)

            sim = cosine_similarity(query_vector, stored_vec)
            if sim >= self.threshold:
                result_bytes = r.hget(_KEY_RESULTS, entry_id)
                if result_bytes is None:
                    continue

                result = json.loads(result_bytes.decode())
                r.incr(_KEY_HITS)

                query_text = (r.hget(_KEY_META, entry_id) or b"").decode()
                logger.info(
                    f"[CACHE] Redis HIT (similarity={sim:.4f}) "
                    f"for '{query_text[:60]}'"
                )
                result["cache_hit"] = True
                result["cache_similarity"] = round(sim, 4)
                return result

        return None

    def _get_fallback(self, query_vector: np.ndarray) -> dict | None:
        self._fallback_queries += 1

        for entry in self._fallback_entries:
            sim = cosine_similarity(query_vector, entry["vector"])
            if sim >= self.threshold:
                self._fallback_hits += 1
                logger.info(
                    f"[CACHE] Memory HIT (similarity={sim:.4f}) "
                    f"for '{entry['query_text'][:60]}'"
                )
                result = dict(entry["result"])
                result["cache_hit"] = True
                result["cache_similarity"] = round(sim, 4)
                return result

        return None

    # ── SET ────────────────────────────────────────────────────────────────

    def set(self, query_text: str, query_vector: np.ndarray, result: dict) -> None:
        """
        Store a query result in the cache.

        Args:
            query_text: Original query string (for logging/meta).
            query_vector: Embedding of the query.
            result: Full pipeline result dict to cache.
        """
        r = _get_redis()

        if r is not None:
            self._set_redis(r, query_text, query_vector, result)
        else:
            self._set_fallback(query_text, query_vector, result)

    def _set_redis(self, r, query_text: str, query_vector: np.ndarray, result: dict) -> None:
        entry_id = str(uuid.uuid4())
        vec_bytes = query_vector.astype(np.float32).tobytes()
        result_json = json.dumps(result)

        pipe = r.pipeline()
        pipe.hset(_KEY_VECTORS, entry_id, vec_bytes)
        pipe.hset(_KEY_RESULTS, entry_id, result_json)
        pipe.hset(_KEY_META, entry_id, query_text)
        # Refresh TTL on every write so active caches don't expire mid-use
        pipe.expire(_KEY_VECTORS, _CACHE_TTL)
        pipe.expire(_KEY_RESULTS, _CACHE_TTL)
        pipe.expire(_KEY_META, _CACHE_TTL)
        pipe.execute()

        logger.debug(
            f"[CACHE] Redis SET '{query_text[:60]}' "
            f"(entry_id={entry_id[:8]})"
        )

    def _set_fallback(self, query_text: str, query_vector: np.ndarray, result: dict) -> None:
        self._fallback_entries.append({
            "query_text": query_text,
            "vector": query_vector.copy(),
            "result": result,
            "timestamp": time.time(),
        })
        logger.debug(
            f"[CACHE] Memory SET '{query_text[:60]}' "
            f"(cache size: {len(self._fallback_entries)})"
        )

    # ── CLEAR ──────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all cached entries (e.g. when documents change)."""
        r = _get_redis()
        if r is not None:
            r.delete(_KEY_VECTORS, _KEY_RESULTS, _KEY_META, _KEY_HITS, _KEY_QUERIES)
            logger.info("[CACHE] Redis cache cleared")
        else:
            self._fallback_entries.clear()
            self._fallback_hits = 0
            self._fallback_queries = 0
            logger.info("[CACHE] Memory cache cleared")

    # ── STATS ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        r = _get_redis()
        if r is not None:
            total = int(r.get(_KEY_QUERIES) or 0)
            hits = int(r.get(_KEY_HITS) or 0)
            entries = r.hlen(_KEY_VECTORS)
            hit_rate = round(hits / total, 3) if total > 0 else 0.0
            return {
                "backend": "redis",
                "total_queries": total,
                "cache_hits": hits,
                "hit_rate": hit_rate,
                "entries_stored": entries,
            }
        else:
            total = self._fallback_queries
            hits = self._fallback_hits
            return {
                "backend": "memory",
                "total_queries": total,
                "cache_hits": hits,
                "hit_rate": round(hits / total, 3) if total > 0 else 0.0,
                "entries_stored": len(self._fallback_entries),
            }


# Module-level singleton
_cache = SemanticCache()


def get_cache() -> SemanticCache:
    """Return the global cache instance."""
    return _cache