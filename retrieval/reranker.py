# cross-encoder to rerank results
from typing import List, Tuple


def _get_reranker_model():
    from sentence_transformers import CrossEncoder
    from config import RERANKER_MODEL_NAME, DEVICE, USE_GPU
    try:
        model = CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)
        if USE_GPU:
            print(f"[Reranker] Using GPU: {DEVICE}")
        else:
            print(f"[Reranker] Using CPU")
    except TypeError:
        # fallback for older versions
        model = CrossEncoder(RERANKER_MODEL_NAME)
        if USE_GPU:
            try:
                import torch
                model.model = model.model.to(DEVICE)
                print(f"[Reranker] Using GPU: {DEVICE}")
            except Exception:
                print(f"[Reranker] GPU setup failed, using CPU")
        else:
            print(f"[Reranker] Using CPU")
    return model


class Reranker:

    def __init__(self):
        self._model = None

    def _model_load(self):
        if self._model is None:
            self._model = _get_reranker_model()

    def rerank(
        self,
        query: str,
        chunk_metadatas: List[dict],
        chunk_texts: List[str],
        top_k: int = 8,
    ) -> List[Tuple[dict, float]]:
        if not chunk_metadatas:
            return []
        self._model_load()
        pairs = [(query, t) for t in chunk_texts]
        scores = self._model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        indexed = list(zip(chunk_metadatas, scores))
        indexed.sort(key=lambda x: -x[1])
        return indexed[:top_k]
