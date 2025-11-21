# utils/retriever.py
import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss

from config import config
from models.embeddings import embed_texts
import re


# -----------------------
# LOAD INDEX & METADATA
# -----------------------
def load_index_and_meta() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    if not os.path.exists(config.VECTOR_STORE_PATH):
        raise FileNotFoundError("FAISS index missing. Build index first.")

    if not os.path.exists(config.METADATA_PATH):
        raise FileNotFoundError("Metadata missing. Build index first.")

    index = faiss.read_index(config.VECTOR_STORE_PATH)

    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


# -----------------------
# SIMPLE LEXICAL SCORE
# -----------------------
def _keyword_overlap(query: str, text: str) -> int:
    """
    Count overlapping unique words between query and text.
    Used as a lexical bonus on top of semantic similarity.
    """
    q_words = set(re.findall(r"\w+", query.lower()))
    t_words = set(re.findall(r"\w+", text.lower()))
    q_words = {w for w in q_words if len(w) > 2}
    t_words = {w for w in t_words if len(w) > 2}
    return len(q_words & t_words)


# -----------------------
# RETRIEVE TOP-K CHUNKS
# -----------------------
def retrieve(
    query: str,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    k: int = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks using:
      1) FAISS semantic similarity
      2) Lexical overlap re-ranking
    Returns list of:
      { score, semantic_score, lexical_score, doc_id, chunk_id, text }
    """
    if not query or not query.strip():
        return []

    k = k or config.MAX_RETRIEVALS

    # --- 1. Embed query and search with FAISS ---
    q_emb = embed_texts([query])[0]
    q_arr = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_arr)

    # Fetch more than k to allow better re-ranking
    search_k = max(k * 2, 12)

    distances, indices = index.search(q_arr, search_k)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue

        m = metadata[idx]
        # L2 distance -> similarity-like score (smaller dist => higher score)
        semantic_score = 1.0 / (1.0 + float(dist))

        candidates.append({
            "semantic_score": semantic_score,
            "doc_id": m["doc_id"],
            "chunk_id": m["chunk_id"],
            "text": m["text"],
        })

    if not candidates:
        return []

    # --- 2. Compute lexical overlap for all candidates ---
    for c in candidates:
        c["lexical_score"] = _keyword_overlap(query, c["text"])

    max_lex = max(c["lexical_score"] for c in candidates) or 1

    # --- 3. Combine scores ---
    LEXICAL_WEIGHT = 0.3  # tune if needed

    for c in candidates:
        lexical_norm = c["lexical_score"] / max_lex
        c["score"] = c["semantic_score"] + LEXICAL_WEIGHT * lexical_norm

    # --- 4. Sort and keep top-k ---
    candidates.sort(key=lambda x: x["score"], reverse=True)

    top = candidates[:k]

    # For UI backwards compatibility, keep only required keys + extra debug
    results: List[Dict[str, Any]] = []
    for c in top:
        results.append({
            "score": c["score"],
            "semantic_score": c["semantic_score"],
            "lexical_score": c["lexical_score"],
            "doc_id": c["doc_id"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
        })

    return results
