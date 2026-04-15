"""
Embedder + FAISS Indexer
Embeds Urdu text chunks using a multilingual sentence-transformer
and stores them in a FAISS index alongside a JSON metadata sidecar.
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to backend/)
# ---------------------------------------------------------------------------
STORAGE_DIR   = Path("storage")
FAISS_PATH    = STORAGE_DIR / "faiss.index"
METADATA_PATH = STORAGE_DIR / "metadata.json"
BM25_PATH     = STORAGE_DIR / "bm25.pkl"

STORAGE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of strings. For multilingual-e5 models the query/passage
    prefix must be added. We use the passage prefix for indexing.
    """
    model = get_embedding_model()
    model_name = os.getenv("EMBEDDING_MODEL", "")

    if "e5" in model_name.lower():
        # multilingual-e5 requires "passage: " prefix at index time
        prefixed = ["passage: " + t for t in texts]
    else:
        prefixed = texts

    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,   # cosine similarity via inner product
        show_progress_bar=len(texts) > 50,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string with the correct prefix."""
    model = get_embedding_model()
    model_name = os.getenv("EMBEDDING_MODEL", "")

    if "e5" in model_name.lower():
        prefixed = "query: " + query
    else:
        prefixed = query

    embedding = model.encode(
        [prefixed],
        normalize_embeddings=True,
    )
    return np.array(embedding, dtype=np.float32)


# ---------------------------------------------------------------------------
# FAISS index management
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product (cosine) index."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_faiss_index(index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, str(FAISS_PATH))
    logger.info(f"FAISS index saved → {FAISS_PATH}")


def load_faiss_index() -> Optional[faiss.IndexFlatIP]:
    if not FAISS_PATH.exists():
        return None
    index = faiss.read_index(str(FAISS_PATH))
    logger.info(f"FAISS index loaded: {index.ntotal} vectors")
    return index


# ---------------------------------------------------------------------------
# Metadata sidecar (JSON)
# ---------------------------------------------------------------------------

def save_metadata(chunks: list[Chunk]) -> None:
    data = [c.to_dict() for c in chunks]
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Metadata saved: {len(data)} chunks → {METADATA_PATH}")


def load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def _tokenize_urdu(text: str) -> list[str]:
    """Simple whitespace tokenizer for Urdu BM25."""
    return text.split()


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    tokenized = [_tokenize_urdu(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    logger.info(f"BM25 index built: {len(tokenized)} documents")
    return bm25


def save_bm25_index(bm25: BM25Okapi) -> None:
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 index saved → {BM25_PATH}")


def load_bm25_index() -> Optional[BM25Okapi]:
    if not BM25_PATH.exists():
        return None
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_chunks(chunks: list[Chunk]) -> dict:
    """
    Full ingestion pipeline:
      1. Embed all chunks
      2. Build + save FAISS index
      3. Build + save BM25 index
      4. Save metadata sidecar
    Returns stats dict.
    """
    if not chunks:
        raise ValueError("No chunks provided for ingestion.")

    logger.info(f"Starting ingestion of {len(chunks)} chunks...")

    texts = [c.text for c in chunks]

    # Dense embeddings
    logger.info("Embedding chunks...")
    embeddings = embed_texts(texts)

    # FAISS
    index = build_faiss_index(embeddings)
    save_faiss_index(index)

    # BM25
    bm25 = build_bm25_index(chunks)
    save_bm25_index(bm25)

    # Metadata
    save_metadata(chunks)

    stats = {
        "chunks_indexed": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "faiss_total": index.ntotal,
    }
    logger.info(f"Ingestion complete: {stats}")
    return stats
