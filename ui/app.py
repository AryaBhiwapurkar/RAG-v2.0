"""
ui/app.py — Gradio frontend for RAG V2.

WHY GRADIO CALLS FASTAPI (not core modules directly):
  Clean separation of concerns:
    - Gradio is a UI layer — knows nothing about retrieval logic
    - FastAPI is the backend — can be tested independently
    - UI can be replaced (e.g. React) without touching backend
    - This is exactly the architecture you defend in interviews

TABS:
  1. Upload (Single) — Upload one PDF, see ingestion status
  2. Upload Bulk — Upload multiple PDFs, ingest IN PARALLEL
  3. Ask — Query interface with faithfulness flag + latency display
  4. Metrics — Live dashboard (P95 latency, cache hit rate)

SESSION NOTE (HF Spaces deployment):
  A banner reminds users that indexes are session-based.
  Documents persist until server restart. Re-upload after refresh.

PHASE 1: BULK UPLOAD
  - New "Upload Bulk" tab using Gradio file_count="multiple"
  - Calls new /ingest-bulk endpoint which runs ThreadPoolExecutor
  - Max 4 concurrent ingestions (configurable)

Run with:
  python ui/app.py
(Make sure FastAPI is running on port 8001 first)
"""

import gradio as gr
import httpx
import time

API_BASE = "http://localhost:8001/api/v2"


# ── API HELPERS ────────────────────────────────────────────────────────────────

def _get(path: str) -> dict:
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post_json(path: str, data: dict) -> dict:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=data, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post_file(path: str, file_path: str, filename: str) -> dict:
    try:
        with open(file_path, "rb") as f:
            r = httpx.post(
                f"{API_BASE}{path}",
                files={"file": (filename, f, "application/pdf")},
                timeout=60,
            )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post_files_bulk(file_objs) -> dict:
    try:
        if not isinstance(file_objs, list):
            file_objs = [file_objs]
        
        file_handles = []
        files = []
        
        for f in file_objs:
            path = f if isinstance(f, str) else f.name
            filename = path.split("/")[-1]
            fh = open(path, "rb")
            file_handles.append(fh)
            files.append(("files", (filename, fh, "application/pdf")))
        
        r = httpx.post(
            f"{API_BASE}/ingest-bulk",
            files=files,
            timeout=120,
        )
        
        for fh in file_handles:
            fh.close()
            
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def upload_pdf(file_obj) -> tuple[str, str]:
    """Handle single PDF upload and return status message + updated doc list."""
    if file_obj is None:
        return "⚠️ Please select a PDF file.", get_document_list()

    filename = file_obj.name.split("/")[-1]
    result = _post_file("/ingest", file_obj.name, filename)

    if "error" in result:
        return f"❌ Upload failed: {result['error']}", get_document_list()

    return (
        f"✅ '{filename}' uploaded. Processing in background...\n"
        "Refresh the document list in a few seconds to check status.",
        get_document_list(),
    )


# ── UPLOAD TAB (Bulk) ──────────────────────────────────────────────────────────
def upload_pdfs_bulk(file_objs) -> tuple[str, str]:
    if not file_objs:
        return "⚠️ Please select PDF files.", get_document_list()

    if not isinstance(file_objs, list):
        file_objs = [file_objs]

    file_paths = [f if isinstance(f, str) else f.name for f in file_objs]
    filenames = [p.split("/")[-1] for p in file_paths]

    result = _post_files_bulk(file_objs)

    if "error" in result:
        return f"❌ Bulk upload failed: {result['error']}", get_document_list()

    return (
        f"✅ {len(filenames)} files uploaded and queued for parallel ingestion.\n"
        "Ingesting up to 4 files concurrently...\n"
        "Refresh the document list in 10-30 seconds to check status.",
        get_document_list(),
    )

def get_document_list() -> str:
    """Return a formatted table of all ingested documents."""
    result = _get("/documents")
    if "error" in result:
        return f"❌ Could not load documents: {result['error']}"

    docs = result.get("documents", [])
    if not docs:
        return "No documents ingested yet. Upload a PDF above."

    lines = ["| # | Filename | Status | Pages | Chunks |",
             "|---|----------|--------|-------|--------|"]
    for i, doc in enumerate(docs, 1):
        status_icon = {"ready": "✅", "processing": "⏳", "failed": "❌"}.get(
            doc["status"], "?"
        )
        lines.append(
            f"| {i} | {doc['filename']} | {status_icon} {doc['status']} "
            f"| {doc.get('pages', '-')} | {doc.get('small_chunks', '-')} |"
        )

    return "\n".join(lines)


# ── QUERY TAB ──────────────────────────────────────────────────────────────────

def ask_question(question: str, doc_ids_str: str) -> tuple[str, str, str]:
    """
    Run a query and return (answer, metadata, confidence_message).
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    # Parse comma-separated doc IDs (empty = search all)
    doc_ids = None
    if doc_ids_str.strip():
        doc_ids = [d.strip() for d in doc_ids_str.split(",") if d.strip()]

    result = _post_json("/query", {"question": question, "doc_ids": doc_ids})

    if "error" in result:
        return f"❌ Query failed: {result['error']}", "", ""

    answer = result.get("answer", "No answer returned.")

    # Faithfulness display
    flag = result.get("faithfulness_flag", "high")
    confidence = result.get("faithfulness_confidence", 1.0)
    flag_display = {
        "high": f"🟢 High confidence ({confidence:.0%})",
        "medium": f"🟡 Medium confidence ({confidence:.0%}) — verify against source",
        "low": f"🔴 Low confidence ({confidence:.0%}) — check document directly",
    }.get(flag, "")

    # Metadata
    meta_lines = [
        f"**Sources:** {', '.join(result.get('sources', []))}",
        f"**Cache hit:** {'Yes ⚡' if result.get('cache_hit') else 'No'}",
        f"**Retrieval:** {result.get('retrieval_latency_ms', 0):.0f}ms",
        f"**Generation:** {result.get('generation_latency_ms', 0):.0f}ms",
        f"**Total:** {result.get('total_latency_ms', 0):.0f}ms",
        f"**Tokens used:** {result.get('token_usage', {}).get('total_tokens', 0)}",
        f"**Context chunks:** {result.get('context_chunks_used', 0)}",
    ]
    metadata = "\n".join(meta_lines)

    return answer, metadata, flag_display


# ── METRICS TAB ───────────────────────────────────────────────────────────────

def get_metrics_display() -> str:
    """Return formatted metrics dashboard."""
    result = _get("/metrics")
    if "error" in result:
        return f"❌ Could not load metrics: {result['error']}"

    p95_status = "✅ Meets target (<2000ms)" if result.get("meets_p95_target") else "❌ Exceeds target"
    ret_status = "✅ Meets target (<200ms)" if result.get("meets_retrieval_target") else "❌ Exceeds target"

    return f"""### Latency (last {result.get('total_queries', 0)} queries)

| Metric | P50 | P95 | P99 | Target |
|--------|-----|-----|-----|--------|
| Total | {result.get('total_p50_ms', 0):.0f}ms | {result.get('total_p95_ms', 0):.0f}ms | {result.get('total_p99_ms', 0):.0f}ms | <2000ms {p95_status} |
| Retrieval | {result.get('retrieval_p50_ms', 0):.0f}ms | {result.get('retrieval_p95_ms', 0):.0f}ms | {result.get('retrieval_p99_ms', 0):.0f}ms | <200ms {ret_status} |

### Cache
- Hit rate: **{result.get('cache_hit_rate', 0):.1%}** (target >30%)
- Entries stored: **{result.get('cache_entries', 0)}**
- Total queries through cache: **{result.get('cache_total_queries', 0)}**

### Tokens
- Avg tokens per query: **{result.get('avg_tokens_per_query', 0)}**
"""


# ── GRADIO APP ─────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="RAG V2 — Document Q&A", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
# 📄 RAG V2 — Production Document Q&A
**Hybrid retrieval** (BM25 + FAISS + RRF + cross-encoder rerank) · **Groq Llama 3.3 70B** · **RAGAS evaluated**

> ⚠️ **Session-based:** Documents persist until server restart. Re-upload after page refresh.
        """)

        with gr.Tabs():

            # ── Tab 1: Upload (Single) ───────────────────────────────────
            with gr.Tab("📁 Upload Single"):
                gr.Markdown("Upload a single text-based PDF to ingest it into the system.")

                with gr.Row():
                    file_input = gr.File(label="Select PDF", file_types=[".pdf"])
                    upload_btn = gr.Button("Upload & Ingest", variant="primary")

                upload_status = gr.Markdown("Ready to upload.")

                gr.Markdown("### Ingested Documents")
                refresh_btn = gr.Button("🔄 Refresh Document List")
                doc_list = gr.Markdown(get_document_list())

                upload_btn.click(
                    fn=upload_pdf,
                    inputs=[file_input],
                    outputs=[upload_status, doc_list],
                )
                refresh_btn.click(fn=get_document_list, outputs=[doc_list])

            # ── Tab 2: Upload (Bulk) ────────────────────────────────────
            with gr.Tab("📂 Upload Bulk (Parallel)"):
                gr.Markdown(
                    "Upload multiple PDFs and ingest them in **parallel** "
                    "(up to 4 concurrent).\n\n"
                    "**Time savings:**\n"
                    "- 5 PDFs sequential: 40s\n"
                    "- 5 PDFs parallel: ~10s"
                )

                with gr.Row():
                    file_input_bulk = gr.File(
                        label="Select PDFs",
                        file_types=[".pdf"],
                        file_count="multiple"  # ← Allows multi-select
                    )
                    upload_btn_bulk = gr.Button("Upload & Ingest (Parallel)", variant="primary")

                upload_status_bulk = gr.Markdown("Ready to upload.")

                gr.Markdown("### Ingested Documents")
                refresh_btn_bulk = gr.Button("🔄 Refresh Document List")
                doc_list_bulk = gr.Markdown(get_document_list())

                upload_btn_bulk.click(
                    fn=upload_pdfs_bulk,
                    inputs=[file_input_bulk],
                    outputs=[upload_status_bulk, doc_list_bulk],
                )
                refresh_btn_bulk.click(fn=get_document_list, outputs=[doc_list_bulk])

            # ── Tab 3: Ask ──────────────────────────────────────────────
            with gr.Tab("💬 Ask a Question"):
                gr.Markdown("Ask anything about your uploaded documents.")

                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="What is the punishment for murder under IPC?",
                    lines=2,
                )
                doc_ids_input = gr.Textbox(
                    label="Document IDs (optional, comma-separated)",
                    placeholder="Leave blank to search all documents",
                )
                ask_btn = gr.Button("Ask", variant="primary")

                with gr.Row():
                    with gr.Column(scale=3):
                        answer_output = gr.Markdown(label="Answer")
                    with gr.Column(scale=1):
                        confidence_output = gr.Markdown(label="Confidence")

                metadata_output = gr.Markdown(label="Query Metadata")

                ask_btn.click(
                    fn=ask_question,
                    inputs=[question_input, doc_ids_input],
                    outputs=[answer_output, metadata_output, confidence_output],
                )
                question_input.submit(
                    fn=ask_question,
                    inputs=[question_input, doc_ids_input],
                    outputs=[answer_output, metadata_output, confidence_output],
                )

            # ── Tab 4: Metrics ──────────────────────────────────────────
            with gr.Tab("📊 Metrics"):
                gr.Markdown("Live latency and cache statistics.")
                metrics_refresh_btn = gr.Button("🔄 Refresh Metrics")
                metrics_display = gr.Markdown(get_metrics_display())
                metrics_refresh_btn.click(fn=get_metrics_display, outputs=[metrics_display])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)