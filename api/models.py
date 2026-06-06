"""
api/models.py — Pydantic request/response schemas for all API endpoints.

WHY PYDANTIC MODELS:
  - FastAPI validates incoming requests automatically (returns 422 on bad input)
  - Response models document exactly what the API returns
  - Type hints throughout = fewer runtime errors
  - Auto-generates OpenAPI docs at /docs (free with FastAPI)

PHASE 1: Added BulkIngestResponse for /ingest-bulk endpoint
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── REQUEST MODELS ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="User's question")
    doc_ids: Optional[list[str]] = Field(
        default=None,
        description="Specific document IDs to search. "
                    "If null, searches all ingested documents."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the punishment for murder under IPC?",
                "doc_ids": None
            }
        }


# ── RESPONSE MODELS ────────────────────────────────────────────────────────────

class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    cache_hit: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    token_usage: TokenUsage
    faithfulness_flag: str          # "high" | "medium" | "low"
    faithfulness_confidence: float  # 0.0–1.0
    context_chunks_used: int
    # UI display hint based on faithfulness
    confidence_message: Optional[str] = None


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    status: str                     # "processing" | "ready" | "failed"
    message: str
    # Filled once processing completes:
    pages: Optional[int] = None
    small_chunks: Optional[int] = None
    large_chunks: Optional[int] = None
    ingestion_time_s: Optional[float] = None
    error: Optional[str] = None


class BulkIngestResponse(BaseModel):
    """
    PHASE 1: Response for bulk multi-file ingestion endpoint.
    
    Clients use this to confirm files are being processed in parallel.
    Poll GET /documents to check individual status.
    """
    files_count: int
    status: str  # "processing"
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "files_count": 5,
                "status": "processing",
                "message": "5 files queued for parallel ingestion (max 4 concurrent). Poll GET /documents to check when all status='ready'."
            }
        }


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    status: str
    pages: Optional[int] = None
    small_chunks: Optional[int] = None
    large_chunks: Optional[int] = None
    ingested_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class MetricsResponse(BaseModel):
    # Latency stats
    total_queries: int
    total_p50_ms: float
    total_p95_ms: float
    total_p99_ms: float
    retrieval_p50_ms: float
    retrieval_p95_ms: float
    retrieval_p99_ms: float
    meets_p95_target: bool
    meets_retrieval_target: bool
    # Cache stats
    cache_hit_rate: float
    cache_entries: int
    cache_total_queries: int
    # Avg tokens
    avg_tokens_per_query: int