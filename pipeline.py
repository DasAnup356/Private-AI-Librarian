# query -> rewrite -> retrieve -> rerank -> generate
import time
from pathlib import Path
from typing import List, Optional, Tuple

from config import (
    INDEX_DIR,
    TOP_K_DENSE,
    TOP_K_SPARSE,
    TOP_K_AFTER_RERANK,
)
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.hybrid import HybridRetriever, dense_only_search
from retrieval.reranker import Reranker
from retrieval.docstore import load_docstore
from llm.query_rewriter import rewrite_query
from llm.answer_generator import generate_answer, build_context_from_chunks
from evaluation.faithfulness import compute_confidence


def _get_retrievers(index_dir: Optional[Path] = None):
    index_dir = Path(index_dir or INDEX_DIR)
    chunk_texts, chunk_metadatas = load_docstore(base_dir=index_dir)
    if not chunk_texts:
        return None, None, None, [], []

    dense = DenseRetriever(index_dir=index_dir)
    dense.load_index(chunk_metadatas)
    sparse = SparseRetriever(index_dir=index_dir)
    sparse.load_index(chunk_metadatas)
    hybrid = HybridRetriever(dense=dense, sparse=sparse)
    reranker = Reranker()
    return dense, sparse, hybrid, reranker, chunk_texts


def run_pipeline(
    query: str,
    *,
    use_hybrid: bool = True,
    use_query_rewrite: bool = True,
    index_dir: Optional[Path] = None,
) -> dict:
    index_dir = Path(index_dir or INDEX_DIR)
    dense, sparse, hybrid, reranker, docstore_texts = _get_retrievers(index_dir)
    if dense is None:
        return {
            "answer": "",
            "context_chunks": [],
            "confidence": 0.0,
            "retrieval_time_s": 0.0,
            "generation_time_s": 0.0,
            "retrieved_chunks": [],
            "rewritten_query": query,
            "error": "No index found. Index documents first.",
        }

    if use_query_rewrite:
        try:
            rewritten = rewrite_query(query)
        except Exception:
            rewritten = query
    else:
        rewritten = query

    t0 = time.perf_counter()
    if use_hybrid:
        merged = hybrid.search(
            rewritten,
            top_k_dense=TOP_K_DENSE,
            top_k_sparse=TOP_K_SPARSE,
        )
    else:
        merged = dense_only_search(rewritten, dense, top_k=TOP_K_DENSE)

    metas = [m for m, _ in merged[: TOP_K_AFTER_RERANK * 2]]
    texts = []
    for m in metas:
        idx = m.get("global_index", -1)
        texts.append(docstore_texts[idx] if 0 <= idx < len(docstore_texts) else "")
    
    reranked = reranker.rerank(rewritten, metas, texts, top_k=TOP_K_AFTER_RERANK)
    retrieval_time_s = time.perf_counter() - t0

    context_texts = []
    retrieved_chunks = []
    for m, score in reranked:
        idx = m.get("global_index", -1)
        t = docstore_texts[idx] if 0 <= idx < len(docstore_texts) else ""
        context_texts.append(t)
        retrieved_chunks.append({"text": t, "score": float(score), "metadata": m})

    context = build_context_from_chunks(context_texts)

    t1 = time.perf_counter()
    try:
        from llm.client import get_llm_client
        client = get_llm_client()
        answer = generate_answer(query, context, client=client)
    except FileNotFoundError as e:
        answer = f"[LLM not loaded: {e}. Place a GGUF model in models/.]"
    except Exception as e:
        answer = f"[Generation error: {e}]"
    generation_time_s = time.perf_counter() - t1

    confidence = compute_confidence(answer, context_texts)

    return {
        "answer": answer,
        "context_chunks": context_texts,
        "confidence": confidence,
        "retrieval_time_s": retrieval_time_s,
        "generation_time_s": generation_time_s,
        "retrieved_chunks": retrieved_chunks,
        "rewritten_query": rewritten,
        "error": None,
    }
