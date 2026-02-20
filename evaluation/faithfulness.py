# score how much answer matches the context
from typing import List

import numpy as np


def _embed(texts: List[str]):
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL_NAME, DEVICE
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    return model.encode(texts, normalize_embeddings=True)


def answer_context_similarity(answer: str, context_chunks: List[str]) -> float:
    # compute max cosine similarity between answer and context chunks
    # returns 0-1 score (higher = more faithful to context)
    if not answer or not context_chunks:
        return 0.0
    a_emb = _embed([answer])
    c_embs = _embed(context_chunks)
    sims = np.dot(c_embs, a_emb.T).flatten()
    max_sim = float(np.max(sims))
    # map cosine [-1,1] to [0,1]
    return (max_sim + 1.0) / 2.0


def compute_confidence(
    answer: str,
    context_chunks: List[str],
    method: str = "max_similarity",
) -> float:
    if method == "max_similarity":
        return answer_context_similarity(answer, context_chunks)
    return 0.0
