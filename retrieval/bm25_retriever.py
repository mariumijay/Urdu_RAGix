"""
Sparse BM25 Retriever
Keyword-based retrieval over tokenized Urdu chunks.
"""

import logging
import numpy as np
from typing import Optional

from rank_bm25 import BM25Okapi

from ingestion.embedder import load_bm25_index, load_metadata
from ingestion.cleaner import normalize_for_search

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    return normalize_for_search(text).split()


class BM25Retriever:
    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._metadata: Optional[list[dict]] = None

    def load(self) -> bool:
        self._bm25     = load_bm25_index()
        self._metadata = load_metadata()
        if self._bm25 is None or not self._metadata:
            logger.warning("BM25 index or metadata not found.")
            return False
        logger.info("BM25Retriever ready")
        return True

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self._metadata)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        BM25 keyword search.
        Returns list of dicts with chunk metadata + score + rank.
        """
        if not self.is_ready:
            raise RuntimeError("BM25Retriever not loaded. Call load() first.")

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                continue
            meta = self._metadata[idx].copy()
            meta["score"]  = score
            meta["rank"]   = rank
            meta["source"] = "bm25"
            results.append(meta)

        return results
