"""
main.py — FastAPI application entry point.

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

API docs auto-available at:
  http://localhost:8000/docs      (Swagger UI)
  http://localhost:8000/redoc     (ReDoc)
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG V2 API",
    description=(
        "Production-grade document Q&A system. "
        "Hybrid retrieval (BM25 + FAISS + RRF + rerank), "
        "Groq Llama 3.3 70B, RAGAS evaluation."
    ),
    version="2.0.0",
)

# CORS — allows Gradio UI (running on different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes under /api/v2
app.include_router(router, prefix="/api/v2")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
