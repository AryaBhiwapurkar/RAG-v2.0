import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from config.settings import settings

logger = logging.getLogger(__name__)

_model = None

def _get_model():
    global _model
    if _model is None:
        logger.info(f"Loading local embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
    return _model

def embed_text(text: str) -> np.ndarray:
    model = _get_model()
    vector = model.encode(text.strip(), normalize_embeddings=True)
    return vector.astype(np.float32)

def embed_batch(texts: list[str]) -> np.ndarray:
    model = _get_model()
    cleaned = [t.strip() if t.strip() else " " for t in texts]
    vectors = model.encode(cleaned, normalize_embeddings=True, show_progress_bar=True)
    result = np.array(vectors, dtype=np.float32)
    logger.info(f"Embedded {len(texts)} texts → shape {result.shape}")
    return result

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))