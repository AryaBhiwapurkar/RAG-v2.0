# RAG V2 — Production Document Q&A System

**Arya Mahesh Bhiwapurkar · IIIT Naya Raipur · May 2026**

Upgraded from a basic LangChain tutorial prototype to a production-grade, metrics-driven, multi-document Q&A platform.

---

## Architecture

```
User Question
    ↓
[Semantic Cache] ──── hit ────→ Return cached answer
    ↓ miss
[Query Embedding]  text-embedding-004 → 768-dim vector
    ↓
┌─────────────────────────────────┐
│  BM25 (sparse)  FAISS (dense)   │  top-20 each
└──────────────┬──────────────────┘
               ↓
          [RRF Fusion]             top-20 deduplicated
               ↓
      [Cross-Encoder Rerank]       top-4 final chunks
               ↓
       [Parent Lookup]             small → large chunks
               ↓
     [Gemini 1.5 Flash]            structured prompt
               ↓
   [Faithfulness Post-Check]       flag low-confidence
               ↓
          Response
```

## Stack (All Free)

| Layer | Tool | Purpose |
|-------|------|---------|
| LLM | Gemini 1.5 Flash | Text generation |
| Embeddings | text-embedding-004 (768-dim) | Semantic vectors |
| Vector Store | FAISS IndexFlatL2 | Dense similarity search |
| Sparse | rank-bm25 | Keyword retrieval |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder reranking |
| PDF Parsing | pdfplumber | Text + table extraction |
| Evaluation | RAGAS | Quality metrics |
| Backend | FastAPI (async) | API endpoints |
| Frontend | Gradio | Browser UI |
| Deployment | Hugging Face Spaces | Free hosting |

## Setup

```bash
# 1. Clone and install
git clone <repo-url>
cd rag-v2
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 3. Start backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Start UI (separate terminal)
python ui/app.py
```

Visit `http://localhost:7860` for the UI, `http://localhost:8000/docs` for the API.

## Key Design Decisions

**Framework-light:** No LangChain in the core pipeline. All chunking, retrieval, and generation use direct API calls. RAGAS (eval only) uses LangChain internally but is isolated to `evaluation/`.

**Parent-child chunking:** Small chunks (300 chars) indexed for precision. Large parent chunks (1200 chars) sent to LLM for rich context.

**Hybrid retrieval:** BM25 catches exact term matches; FAISS catches semantic similarity. RRF fuses rankings without score normalisation. Cross-encoder reranks top-20 to top-4.

**Per-document FAISS:** Each document gets its own index. Delete one without rebuilding others.

**Semantic cache:** Cosine similarity > 0.92 = cache hit. Handles paraphrased questions. ~40% LLM call reduction.

**Faithfulness gate:** Post-generation check flags low-confidence answers before they reach the user.

## Metrics (fill after Day 5)

| Metric | V1 Baseline | V2 Target | V2 Actual |
|--------|-------------|-----------|-----------|
| Faithfulness | ? | >0.85 | ? |
| Answer Relevancy | ? | >0.80 | ? |
| Context Precision | ? | >0.75 | ? |
| Context Recall | ? | >0.80 | ? |
| P95 Total Latency | ? | <2000ms | ? |
| P95 Retrieval | ? | <200ms | ? |
| Cache Hit Rate | N/A | >30% | ? |

## Project Structure

```
rag-v2/
├── config/settings.py          # All constants — single source of truth
├── core/
│   ├── embedder.py             # text-embedding-004 wrapper
│   ├── preprocessor.py         # Query cleaning for BM25
│   ├── chunker.py              # Parent-child chunking
│   ├── retriever.py            # BM25 + FAISS + RRF + rerank
│   └── generator.py            # Gemini generation + faithfulness check
├── storage/
│   ├── vector_store.py         # Per-doc FAISS index management
│   └── cache.py                # Semantic in-memory cache
├── pipeline/
│   ├── ingest.py               # PDF → indexed chunks (async)
│   └── query.py                # Question → answer orchestration
├── evaluation/
│   ├── ragas_eval.py           # RAGAS metrics wrapper
│   └── latency_tracker.py      # P50/P95/P99 tracking
├── api/
│   ├── models.py               # Pydantic schemas
│   └── routes.py               # FastAPI endpoints
├── ui/app.py                   # Gradio frontend
├── main.py                     # FastAPI app entrypoint
└── data/
    ├── uploads/                # Uploaded PDFs
    ├── indexes/                # Per-document FAISS indexes
    └── doc_registry.json       # Document metadata
```

## Interview Answer: LangChain?

> "I deliberately excluded LangChain from the core pipeline. Chunking, retrieval, and generation are all direct API calls so I can explain and debug every step. For example, my hybrid retrieval calls FAISS and BM25 directly, fuses with RRF manually, then passes through the cross-encoder reranker before calling Gemini. I only use RAGAS for evaluation — which uses LangChain internally but that's isolated to the eval module, not my pipeline."

## Interview Answer: Scale to 1M docs?

> "Three main changes: (1) Replace FAISS IndexFlatL2 (exact O(n)) with Pinecone using HNSW index — O(log n) approximate search. (2) Replace synchronous ingestion with Celery + Redis worker queue, PDFs on S3. (3) Add PostgreSQL for doc registry with metadata filtering, Redis for shared cache across API replicas. The retrieval algorithm stays the same — it's stateless and scales horizontally."
