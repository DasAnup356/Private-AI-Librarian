# merge dense + sparse with RRF
from pathlib import Path
from typing import List, Optional, Tuple

from .dense import DenseRetriever
from .sparse import SparseRetriever


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[dict, float]]],
    k: int = 60,
) -> List[Tuple[dict, float]]:
    # RRF: score = sum of 1/(k + rank) across all lists
    # k=60 is standard, lower k = more weight to top results
    from config import TOP_K_AFTER_MERGE
    
    def key(meta: dict) -> tuple:
        return (meta.get("doc_id", ""), meta.get("chunk_index", -1))

    scores: dict = {}
    for rlist in ranked_lists:
        for rank, (meta, _) in enumerate(rlist, start=1):
            kk = key(meta)
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (k + rank)

    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    meta_by_key = {}
    for rlist in ranked_lists:
        for meta, _ in rlist:
            meta_by_key[key(meta)] = meta
    return [(meta_by_key[k], s) for k, s in sorted_items[:TOP_K_AFTER_MERGE]]


class HybridRetriever:
    # combines dense and sparse retrieval, merges with RRF

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
    ) -> List[Tuple[dict, float]]:
        dense_results = self.dense.search(query, top_k=top_k_dense)
        sparse_results = self.sparse.search(query, top_k=top_k_sparse)
        merged = _reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
        )
        return merged


def dense_only_search(
    query: str,
    retriever: DenseRetriever,
    top_k: int,
) -> List[Tuple[dict, float]]:
    return retriever.search(query, top_k=top_k)
