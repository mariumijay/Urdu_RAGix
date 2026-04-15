"""
Hybrid Retrieval via Reciprocal Rank Fusion (RRF)
Merges FAISS (dense) and BM25 (sparse) result lists into a single
ranked list using the RRF formula:
    RRF(d) = Σ  1 / (k + rank(d))    where k = 60
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

RRF_K = 60  # standard constant from the original RRF paper


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    top_n: int = 30,
) -> list[dict]:
    """
    Merge multiple ranked result lists using RRF.

    Args:
        result_lists: Each list is a ranked list of chunk dicts.
                      Each dict must have a "chunk_id" key and a "rank" key.
        top_n:        How many fused results to return.

    Returns:
        Sorted list of chunk dicts with added "rrf_score" key.
    """
    # Map chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = defaultdict(float)
    # Map chunk_id → the chunk dict (keep last seen, they should be identical)
    chunks_by_id: dict[str, dict] = {}

    for result_list in result_lists:
        for item in result_list:
            cid  = item["chunk_id"]
            rank = item.get("rank", 0)   # 0-based rank within this list
            rrf_scores[cid]  += 1.0 / (RRF_K + rank + 1)
            chunks_by_id[cid] = item

    # Sort by descending RRF score
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    fused: list[dict] = []
    for new_rank, cid in enumerate(sorted_ids[:top_n]):
        chunk = chunks_by_id[cid].copy()
        chunk["rrf_score"] = rrf_scores[cid]
        chunk["fused_rank"] = new_rank
        fused.append(chunk)

    logger.info(
        f"RRF fusion: {sum(len(r) for r in result_lists)} candidates → {len(fused)} fused"
    )
    return fused
