# utils/ingest.py
import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from config import config
from models.embeddings import embed_texts

# PDF reading
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Text splitter (LangChain 0.2+)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    raise RuntimeError(
        "langchain-text-splitters is required. Install with:\n\n"
        "    pip install langchain-text-splitters\n"
    ) from e

import numpy as np
import faiss


# ----------------------------
# TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    if PdfReader is None:
        raise RuntimeError("pypdf is required to read PDFs. Install with: pip install pypdf")

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)

    cleaned = []
    for p in pages:
        t = p.replace("\r", "\n")
        t = re.sub(r"\n{2,}", "\n\n", t)
        cleaned.append(t.strip())

    return "\n\n".join(cleaned)


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        t = f.read()
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()


# ----------------------------
# CHUNKING
# ----------------------------
def chunk_text(text: str, doc_id: str) -> List[Dict[str, Any]]:
    """
    Chunk text using RecursiveCharacterTextSplitter.
    Chunk size & overlap are in characters (from config).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,          # e.g. 1200 chars
        chunk_overlap=config.CHUNK_OVERLAP,    # e.g. 200 chars
        separators=["\n\n", "\n", ".", " "],
    )

    parts = splitter.split_text(text)
    chunks: List[Dict[str, Any]] = []

    for i, chunk in enumerate(parts):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunks.append(
            {
                "doc_id": doc_id,
                "chunk_id": i,
                "text": chunk,
            }
        )

    return chunks


# ----------------------------
# MAIN INGEST FUNCTION
# ----------------------------
def index_documents(
    file_paths: List[str],
    save_index: bool = True,
    debug: bool = False,
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Ingest PDF/TXT files, chunk them, embed chunks with HF embeddings,
    build a FAISS L2 index and save index + metadata.

    Returns:
        (faiss_index, metadata_list)
    """
    all_chunks: List[Dict[str, Any]] = []

    for path in file_paths:
        path = str(path)
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            text = extract_text_from_pdf(path)
        else:
            text = extract_text_from_txt(path)

        doc_id = Path(path).name
        chunks = chunk_text(text, doc_id)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks produced from the provided documents.")

    # Texts to embed
    texts = [c["text"] for c in all_chunks]

    # Embed in batches to avoid memory issues
    embeddings: List[List[float]] = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = embed_texts(batch)
        embeddings.extend(embs)

    # Convert to numpy array
    xb = np.array(embeddings, dtype="float32")
    # Normalize for L2 similarity (optional but good practice)
    faiss.normalize_L2(xb)

    dim = xb.shape[1]
    index = faiss.IndexFlatL2(dim)

    # ✅ Correct call: pass ONLY the matrix
    index.add(xb)

    # Save index + metadata
    if save_index:
        data_dir = Path(config.VECTOR_STORE_PATH).parent
        data_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, config.VECTOR_STORE_PATH)

        with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        if debug:
            debug_info = {
                "total_chunks": len(all_chunks),
                "sample_first": all_chunks[0]["text"] if all_chunks else "",
                "sample_last": all_chunks[-1]["text"] if all_chunks else "",
            }
            with open(data_dir / "ingest_debug.json", "w", encoding="utf-8") as df:
                json.dump(debug_info, df, ensure_ascii=False, indent=2)

    return index, all_chunks
