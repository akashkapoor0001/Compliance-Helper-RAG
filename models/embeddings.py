# models/embeddings.py
from typing import List
import threading

# Hugging Face encoder
from sentence_transformers import SentenceTransformer

# ------------------------
# GLOBAL MODEL LOADER (singleton)
# ------------------------
_model_lock = threading.Lock()
# store the model in a one-element list (mutable container) to avoid using the `global` statement
_embedding_model = [None]

def _load_embedding_model():
    """
    Loads the sentence-transformers embedding model once (singleton).
    """
    with _model_lock:
        if _embedding_model[0] is None:
            _embedding_model[0] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _embedding_model[0]
        return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Returns embeddings using HuggingFace sentence-transformers.
    Runs completely local, no API needed.
    """
    if not isinstance(texts, (list, tuple)):
        raise ValueError("embed_texts expects a list of strings")

    model = _load_embedding_model()

    # Encode (convert to numpy array)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True   # normalizes automatically → great for FAISS L2/IP
    )

    # Convert numpy -> Python lists (FAISS expects float32)
    return vectors.tolist()