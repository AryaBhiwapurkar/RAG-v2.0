"""
config/settings.py — Single source of truth for all configuration.

Every constant lives here. Every module imports from here.
No magic numbers anywhere else in the codebase.

pydantic-settings reads from .env automatically.

PHASE 1 CHANGES:
  - embedding_model: all-MiniLM-L6-v2 → paraphrase-MiniLM-L3-v2  (2x faster, same 384-dim)
  - bm25_top_k:  20 → 10   (halves reranker input, cuts ~40% retrieval time)
  - faiss_top_k: 20 → 10   (same reason)
"""

from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────────────────
    groq_api_key: str = "not_set"

    # ── Model Names ───────────────────────────────────────────────────────
    llm_model: str = "llama-3.3-70b-versatile"

    # PHASE 1: L3 vs L6 — 3-layer vs 6-layer transformer.
    # L3 is ~2x faster at encode time with <5% accuracy drop on retrieval tasks.
    # Both output 384-dim vectors so FAISS index shape is unchanged.
    # Existing indexes built with L6 must be re-ingested after this change.
    embedding_model: str = "paraphrase-MiniLM-L3-v2"

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM Generation ────────────────────────────────────────────────────
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 1024

    # ── Chunking ──────────────────────────────────────────────────────────
    small_chunk_size: int = 300
    small_chunk_overlap: int = 50
    large_chunk_size: int = 1200
    large_chunk_overlap: int = 100

    # ── Retrieval ─────────────────────────────────────────────────────────
    # PHASE 1: Reduced from 20 → 10.
    # Cross-encoder rerank is O(n) on CPU — cutting candidates from 20 to 10
    # roughly halves rerank time. RRF still fuses both lists so coverage stays good.
    # If RAGAS recall drops after this change, bump back to 15 as a middle ground.
    bm25_top_k: int = 10
    faiss_top_k: int = 10

    rrf_k_constant: int = 60
    rerank_top_k: int = 4

    # ── Cache ─────────────────────────────────────────────────────────────
    cache_similarity_threshold: float = 0.92

    # ── Faithfulness Post-Check ───────────────────────────────────────────
    faithfulness_confidence_threshold: float = 0.6

    # ── PDF Validation ────────────────────────────────────────────────────
    pdf_min_total_chars: int = 100
    pdf_min_chars_per_page: int = 50
    pdf_max_pages: int = 200


    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0



    # ── Paths ─────────────────────────────────────────────────────────────
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    indexes_dir: Path = BASE_DIR / "data" / "indexes"
    cache_dir: Path = BASE_DIR / "data" / "cache"
    doc_registry_path: Path = BASE_DIR / "data" / "doc_registry.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.indexes_dir.mkdir(parents=True, exist_ok=True)
settings.cache_dir.mkdir(parents=True, exist_ok=True)