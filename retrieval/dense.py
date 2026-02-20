# FAISS + BGE for semantic search
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL_NAME, DEVICE, USE_GPU
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    if USE_GPU:
        print(f"[Embeddings] Using GPU: {DEVICE}")
    else:
        print(f"[Embeddings] Using CPU")
    return model


class DenseRetriever:

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        model_name: Optional[str] = None,
    ):
        from config import INDEX_DIR, EMBEDDING_MODEL_NAME
        self.index_dir = Path(index_dir or INDEX_DIR)
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self._model = None
        self._index = None
        self._id_to_chunk: List[dict] = []

    def _model_load(self):
        if self._model is None:
            self._model = _get_embedding_model()

    def embed(self, texts: List[str]) -> np.ndarray:
        self._model_load()
        emb = self._model.encode(texts, normalize_embeddings=True)
        return np.array(emb, dtype=np.float32)

    def build_index(self, chunk_texts: List[str], chunk_metadatas: List[dict]) -> None:
        import faiss
        self._model_load()
        vectors = self.embed(chunk_texts)
        d = vectors.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
        self._index = index
        self._id_to_chunk = list(chunk_metadatas)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_dir / "faiss.index"))

    def load_index(self, chunk_metadatas: List[dict]) -> bool:
        import faiss
        path = self.index_dir / "faiss.index"
        if not path.exists():
            return False
        self._index = faiss.read_index(str(path))
        self._id_to_chunk = chunk_metadatas
        return True

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[dict, float]]:
        if self._index is None:
            return []
        self._model_load()
        q = self.embed([query])
        scores, indices = self._index.search(q, min(top_k, len(self._id_to_chunk)))
        results = []
        for s, i in zip(scores[0], indices[0]):
            if i < 0:
                break
            results.append((self._id_to_chunk[i], float(s)))
        return results
