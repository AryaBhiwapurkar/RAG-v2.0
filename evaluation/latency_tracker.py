"""
evaluation/latency_tracker.py — Latency and token cost tracking.

WHY PERCENTILES (not averages):
  Average latency hides outliers. P95 = "95% of queries finish under X ms".
  This is how production SLOs are written (e.g. "P95 < 2000ms").
  Your design doc target: P95 < 2000ms total, retrieval < 200ms.

WHAT'S TRACKED:
  - Total query latency (ms)
  - Retrieval-only latency (ms)
  - Token counts (for cost estimation)
  - Query count

In-memory only (resets on server restart). For persistent tracking,
write to a log file or database (V3).
"""

import numpy as np
import logging
from collections import deque
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

# Keep last N queries for percentile calculations
WINDOW_SIZE = 200


@dataclass
class LatencyRecord:
    total_ms: float
    retrieval_ms: float
    timestamp: float = field(default_factory=time.time)


class LatencyTracker:
    """
    Rolling-window latency tracker with percentile support.
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self._records: deque[LatencyRecord] = deque(maxlen=window_size)
        self._total_queries = 0
        self._total_tokens = 0

    def record(self, total_ms: float, retrieval_ms: float, tokens: int = 0) -> None:
        """Record latency for one query."""
        self._records.append(LatencyRecord(total_ms=total_ms, retrieval_ms=retrieval_ms))
        self._total_queries += 1
        self._total_tokens += tokens

    def _percentile(self, values: list[float], p: float) -> float:
        """Compute the p-th percentile of a list of values."""
        if not values:
            return 0.0
        return float(np.percentile(values, p))

    @property
    def stats(self) -> dict:
        """
        Return summary statistics for the tracking window.

        Returns dict with P50/P95/P99 for total and retrieval latency,
        plus overall query count and average tokens.
        """
        if not self._records:
            return {
                "total_queries": self._total_queries,
                "window_size": 0,
                "total_p50_ms": 0,
                "total_p95_ms": 0,
                "total_p99_ms": 0,
                "retrieval_p50_ms": 0,
                "retrieval_p95_ms": 0,
                "retrieval_p99_ms": 0,
                "avg_tokens_per_query": 0,
                "meets_p95_target": False,  # Target: <2000ms
                "meets_retrieval_target": False,  # Target: <200ms
            }

        total_times = [r.total_ms for r in self._records]
        retrieval_times = [r.retrieval_ms for r in self._records]

        total_p95 = self._percentile(total_times, 95)
        retrieval_p95 = self._percentile(retrieval_times, 95)

        return {
            "total_queries": self._total_queries,
            "window_size": len(self._records),
            "total_p50_ms": round(self._percentile(total_times, 50), 1),
            "total_p95_ms": round(total_p95, 1),
            "total_p99_ms": round(self._percentile(total_times, 99), 1),
            "retrieval_p50_ms": round(self._percentile(retrieval_times, 50), 1),
            "retrieval_p95_ms": round(retrieval_p95, 1),
            "retrieval_p99_ms": round(self._percentile(retrieval_times, 99), 1),
            "avg_tokens_per_query": (
                round(self._total_tokens / self._total_queries)
                if self._total_queries > 0 else 0
            ),
            "meets_p95_target": total_p95 < 2000,    # Design doc target
            "meets_retrieval_target": retrieval_p95 < 200,
        }


# Module-level singleton
_tracker = LatencyTracker()


def get_tracker() -> LatencyTracker:
    """Return the global tracker instance."""
    return _tracker
