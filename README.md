# RAG V2 - Document Q&A System

Production-style multi-document Q&A app with PDF ingestion, hybrid retrieval, reranking, semantic cache, latency metrics, FastAPI backend, and Gradio UI.

## Architecture

```text
User Question
    |
[Semantic Cache] -- hit --> Return cached answer
    |
  miss
    |
[Query Embedding] all-MiniLM-L6-v2
    |
[BM25 sparse search] + [FAISS dense search]
    |
[RRF Fusion]
    |
[Cross-Encoder Rerank]
    |
[Parent Chunk Lookup]
    |
[Groq Llama 3.3 70B]
    |
Response + sources + latency/token metadata
```

## Stack

| Layer | Tool |
| --- | --- |
| LLM | Groq, `llama-3.3-70b-versatile` |
| Embeddings | Sentence Transformers, `all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Sparse search | rank-bm25 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF parsing | pdfplumber |
| Backend | FastAPI |
| Frontend | Gradio |
| Evaluation | RAGAS |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY in .env
```

## Run

Start the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Start the UI in another terminal:

```bash
python ui/app.py
```

Open:

- UI: `http://localhost:7860`
- API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

## API

All main routes are under `/api/v2`:

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/api/v2/ingest` | Upload a PDF and start background ingestion |
| GET | `/api/v2/documents` | List ingested documents and statuses |
| POST | `/api/v2/query` | Ask a question over ready documents |
| GET | `/api/v2/metrics` | View latency, cache, and token metrics |

## Project Structure

```text
rag-v2/
├── api/                  # FastAPI request/response models and routes
├── config/settings.py    # Runtime settings and paths
├── core/                 # Chunking, embeddings, retrieval, generation
├── data/                 # Local registry plus ignored uploads/indexes/cache
├── evaluation/           # RAGAS and latency tracking
├── pipeline/             # Ingestion and query orchestration
├── storage/              # FAISS/BM25 persistence and semantic cache
├── ui/app.py             # Gradio frontend
├── main.py               # FastAPI app entrypoint
└── requirements.txt
```

## Notes

- Uploaded PDFs, generated indexes, cache files, `.env`, and virtualenv files are intentionally ignored by Git.
- `data/doc_registry.json` is kept as the local document registry seed.
- RAGAS is isolated to `evaluation/`; the core retrieval/generation pipeline does not depend on LangChain.
