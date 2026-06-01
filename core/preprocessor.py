"""
core/preprocessor.py — Query preprocessing for BM25 only.

Raw user queries are noisy ("can you please tell me what is the punishment
for murder?"). BM25 is keyword-based and case-sensitive, so stripping
filler words and normalising significantly improves keyword matching.

CRITICAL DESIGN DECISION:
  Preprocess for BM25 only.
  Dense embeddings (FAISS) get the ORIGINAL query.
  Reason: embedding models understand natural language — stripping words
  from the semantic query hurts recall. Preprocessing only helps BM25.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Words that carry no retrieval signal for BM25
FILLER_WORDS = {
    "can", "you", "tell", "me", "what", "is", "the", "are", "does",
    "do", "please", "a", "an", "of", "in", "on", "at", "to", "for",
    "how", "why", "when", "where", "who", "which", "was", "were",
    "will", "would", "could", "should", "i", "my", "give", "show",
    "explain", "describe", "find", "get", "list", "summarize",
}

# Common abbreviations to expand for better BM25 term matching
ABBREVIATIONS = {
    "ipc": "indian penal code",
    "sec": "section",
    "art": "article",
    "govt": "government",
    "dept": "department",
    "vs": "versus",
    "max": "maximum",
    "min": "minimum",
    "approx": "approximately",
}


def preprocess_query(query: str) -> str:
    """
    Clean a user query for BM25 keyword retrieval.

    Steps:
      1. Lowercase + strip whitespace
      2. Expand known abbreviations
      3. Remove punctuation (keeps alphanumerics + spaces)
      4. Remove filler words
      5. Collapse extra whitespace

    Falls back to original query if preprocessing empties the string
    (avoids edge case of "what is the?" → "" → BM25 failure).

    Args:
        query: Raw user query string.

    Returns:
        Cleaned string optimised for BM25 keyword matching.
    """
    original = query.strip()
    q = original.lower().strip()

    # Expand abbreviations (whole-word match only)
    for abbr, expansion in ABBREVIATIONS.items():
        q = re.sub(rf"\b{abbr}\b", expansion, q)

    # Remove punctuation, keep alphanumerics and spaces
    q = re.sub(r"[^\w\s]", " ", q)

    # Remove filler words
    tokens = [t for t in q.split() if t not in FILLER_WORDS]

    # Fallback: if preprocessing removed everything, use original
    if not tokens:
        logger.debug(f"Preprocessing emptied query — falling back to original: '{original}'")
        return original

    result = " ".join(tokens)
    logger.debug(f"Preprocessed query: '{original}' → '{result}'")
    return result
