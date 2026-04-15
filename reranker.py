"""
Cross-Encoder Reranker
Reranks hybrid-fused candidates using a cross-encoder model for
precise relevance scoring. Falls back gracefully if model unavailable.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(model_name, max_length=512)
            logger.info(f"Reranker loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            _reranker = None
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank candidate chunks using a cross-encoder.

    Args:
        query:      The user's Urdu query string.
        candidates: List of chunk dicts (output of hybrid fusion).
        top_k:      Number of top chunks to return.

    Returns:
        Top-k reranked chunk dicts with added "rerank_score" key.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    if reranker is None:
        # Graceful fallback: return top_k by rrf_score
        logger.warning("Reranker unavailable — using RRF scores as fallback.")
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("rrf_score", 0),
            reverse=True,
        )
        for i, c in enumerate(sorted_candidates[:top_k]):
            c["rerank_score"] = c.get("rrf_score", 0.0)
            c["final_rank"]   = i
        return sorted_candidates[:top_k]

    # Build query-passage pairs for the cross-encoder
    pairs = [(query, c["text"]) for c in candidates]

    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.error(f"Reranker prediction failed: {e}")
        # Fallback to RRF order
        for i, c in enumerate(candidates[:top_k]):
            c["rerank_score"] = c.get("rrf_score", 0.0)
            c["final_rank"]   = i
        return candidates[:top_k]

    # Attach scores and sort descending
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    for i, chunk in enumerate(reranked[:top_k]):
        chunk["final_rank"] = i

    logger.info(
        f"Reranking: {len(candidates)} candidates → top {top_k} selected "
        f"(best score: {reranked[0]['rerank_score']:.4f})"
    )
    return reranked[:top_k]
