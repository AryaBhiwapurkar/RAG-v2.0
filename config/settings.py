"""
config/settings.py — Single source of truth for all configuration.

Every constant lives here. Every module imports from here.
No magic numbers anywhere else in the codebase.

pydantic-settings reads from .env automatically.
"""

from pydantic_settings import BaseSettings
from pathlib import Path

# Project root: two levels up from this file (rag-v2/)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ── API Keys ──────────────────────────────────────────────────────────
    google_api_key: str = "not_set"
    groq_api_key: str = "not_set"  

    # ── Model Names ───────────────────────────────────────────────────────
    llm_model: str = "models/gemini-2.0-flash"  # Google Gemini Pro (2024-09-26)
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM Generation ────────────────────────────────────────────────────
    llm_temperature: float = 0.1          # Low temp = factual, less creative
    llm_max_output_tokens: int = 1024

    # ── Chunking ──────────────────────────────────────────────────────────
    small_chunk_size: int = 300           # Retrieved for precision
    small_chunk_overlap: int = 50
    large_chunk_size: int = 1200          # Sent to LLM for rich context
    large_chunk_overlap: int = 100

    # ── Retrieval ─────────────────────────────────────────────────────────
    bm25_top_k: int = 20                  # Sparse candidates
    faiss_top_k: int = 20                 # Dense candidates
    rrf_k_constant: int = 60             # RRF formula constant (prevents top-rank dominance)
    rerank_top_k: int = 4                 # Final chunks sent to LLM

    # ── Cache ─────────────────────────────────────────────────────────────
    cache_similarity_threshold: float = 0.92   # Cosine threshold for cache hit

    # ── Faithfulness Post-Check ───────────────────────────────────────────
    faithfulness_confidence_threshold: float = 0.6

    # ── PDF Validation ────────────────────────────────────────────────────
    pdf_min_total_chars: int = 100
    pdf_min_chars_per_page: int = 50
    pdf_max_pages: int = 200

    # ── Paths ─────────────────────────────────────────────────────────────
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    indexes_dir: Path = BASE_DIR / "data" / "indexes"
    cache_dir: Path = BASE_DIR / "data" / "cache"
    doc_registry_path: Path = BASE_DIR / "data" / "doc_registry.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single instance — import this everywhere:
#   from config.settings import settings
settings = Settings()

# Ensure data directories exist at import time
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.indexes_dir.mkdir(parents=True, exist_ok=True)
settings.cache_dir.mkdir(parents=True, exist_ok=True)
