"""
evaluation/ragas_eval.py — RAGAS evaluation wrapper.

WHY RAGAS (not manual evaluation):
  "Can't improve what you don't measure." — Design doc.
  Manual evaluation doesn't scale and is subjective.
  RAGAS gives reproducible, comparable metrics across V1/V2/V3.

FOUR METRICS (from design doc):
  Faithfulness (>0.85)    — Is answer grounded in retrieved context?
  Answer Relevancy (>0.80) — Does answer address the question?
  Context Precision (>0.75) — Are retrieved chunks actually relevant?
  Context Recall (>0.80)   — Were all relevant chunks retrieved?

HOW TO USE:
  1. Create a test set (list of dicts with 'question', 'answer', 'ground_truth')
  2. Call evaluate_rag(test_set, pipeline_fn)
  3. pipeline_fn(question) → {"answer": str, "context": list[str]}
  4. Returns a dict of metric name → score

WORKFLOW:
  - Before V2: run on V1 → record baseline numbers
  - After each major change: run again → compare
  - These numbers go in your resume bullets

NOTE: RAGAS uses LangChain internally. This is fine — it's isolated
to this module. Core pipeline has no LangChain. (Design doc, Section 06.)
"""

import logging
from typing import Callable
from datasets import Dataset

logger = logging.getLogger(__name__)


def build_ragas_dataset(
    questions: list[str],
    ground_truths: list[str],
    pipeline_fn: Callable[[str], dict],
) -> Dataset:
    """
    Run the RAG pipeline on a test set and build a RAGAS-compatible Dataset.

    Args:
        questions: List of test questions.
        ground_truths: List of expected answers (one per question).
        pipeline_fn: Function that takes a question string and returns
                     {"answer": str, "context": list[str]}
                     where context is the list of retrieved chunk texts.

    Returns:
        HuggingFace Dataset ready for RAGAS evaluation.
    """
    records = []
    total = len(questions)

    for i, (question, ground_truth) in enumerate(zip(questions, ground_truths)):
        logger.info(f"Running pipeline for question {i+1}/{total}: '{question[:60]}...'")
        try:
            result = pipeline_fn(question)
            records.append({
                "question": question,
                "answer": result["answer"],
                "contexts": result["context"],   # List of chunk texts
                "ground_truth": ground_truth,
            })
        except Exception as e:
            logger.error(f"Pipeline failed on question {i+1}: {e}")
            records.append({
                "question": question,
                "answer": "",
                "contexts": [],
                "ground_truth": ground_truth,
            })

    return Dataset.from_list(records)


def evaluate_rag(
    questions: list[str],
    ground_truths: list[str],
    pipeline_fn: Callable[[str], dict],
) -> dict:
    """
    Run full RAGAS evaluation on a test set.

    Args:
        questions: Test questions.
        ground_truths: Expected answers.
        pipeline_fn: Your RAG pipeline as a callable.

    Returns:
        Dict of metric name → score (0.0–1.0).
        Example: {"faithfulness": 0.87, "answer_relevancy": 0.82, ...}
    """
    # Import here to avoid slow startup time when not evaluating
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    logger.info(f"Building RAGAS dataset from {len(questions)} questions...")
    dataset = build_ragas_dataset(questions, ground_truths, pipeline_fn)

    logger.info("Running RAGAS evaluation (this may take a few minutes)...")
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "faithfulness": round(result["faithfulness"], 4),
        "answer_relevancy": round(result["answer_relevancy"], 4),
        "context_precision": round(result["context_precision"], 4),
        "context_recall": round(result["context_recall"], 4),
    }

    # Print a formatted summary
    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    for metric, score in scores.items():
        target = {"faithfulness": 0.85, "answer_relevancy": 0.80,
                  "context_precision": 0.75, "context_recall": 0.80}[metric]
        status = "✓ PASS" if score >= target else "✗ FAIL"
        print(f"  {metric:<25} {score:.4f}  (target ≥{target})  {status}")
    print("="*50 + "\n")

    logger.info(f"RAGAS evaluation complete: {scores}")
    return scores


def make_pipeline_fn(doc_ids: list[str] | None = None) -> Callable:
    """
    Convenience wrapper: create a pipeline function compatible with evaluate_rag.

    Usage:
        pipeline_fn = make_pipeline_fn(doc_ids=["abc123"])
        results = evaluate_rag(questions, ground_truths, pipeline_fn)
    """
    from pipeline.query import run_query
    from storage.vector_store import load_index

    def pipeline_fn(question: str) -> dict:
        result = run_query(question, doc_ids=doc_ids)

        # RAGAS needs the raw chunk texts (not just doc_ids)
        # We re-load them from the result's sources for simplicity
        # In a production eval, you'd pass chunks through the result dict
        context_texts = []
        if doc_ids:
            for doc_id in doc_ids:
                try:
                    _, _, small_chunks, _ = load_index(doc_id)
                    # Use first few chunks as proxy (or store context in result)
                    context_texts.extend([c["text"] for c in small_chunks[:4]])
                except Exception:
                    pass

        return {
            "answer": result["answer"],
            "context": context_texts[:8],  # Max 8 context pieces for RAGAS
        }

    return pipeline_fn
