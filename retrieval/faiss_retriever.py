"""
Dense FAISS Retriever
Searches the FAISS index with a query embedding and returns top-k results.
"""

import logging
import numpy as np
from typing import Optional

import faiss

from ingestion.embedder import embed_query, load_faiss_index, load_metadata

logger = logging.getLogger(__name__)


class FAISSRetriever:
    def __init__(self):
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: Optional[list[dict]] = None

    def load(self) -> bool:
        self._index    = load_faiss_index()
        self._metadata = load_metadata()
        if self._index is None or not self._metadata:
            logger.warning("FAISS index or metadata not found.")
            return False
        logger.info(f"FAISSRetriever ready: {self._index.ntotal} vectors")
        return True

    @property
    def is_ready(self) -> bool:
        return self._index is not None and bool(self._metadata)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Search FAISS index with query.
        Returns list of dicts with keys: chunk_id, text, score, rank, + metadata.
        """
        if not self.is_ready:
            raise RuntimeError("FAISSRetriever not loaded. Call load() first.")

        query_vec = embed_query(query)               # shape (1, dim)
        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            meta = self._metadata[idx].copy()
            meta["score"]  = float(score)
            meta["rank"]   = rank
            meta["source"] = "faiss"
            results.append(meta)

        return results
