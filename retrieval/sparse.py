# BM25 keyword search
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return tokens


class SparseRetriever:

    def __init__(self, index_dir: Optional[Path] = None):
        from config import INDEX_DIR
        self.index_dir = Path(index_dir or INDEX_DIR)
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []
        self._id_to_chunk: List[dict] = []

    def build_index(self, chunk_texts: List[str], chunk_metadatas: List[dict]) -> None:
        self._corpus_tokens = [_tokenize(t) for t in chunk_texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._id_to_chunk = list(chunk_metadatas)
        self._save_metadata()

    def _save_metadata(self) -> None:
        # save metadata and tokenized corpus so we don't have to retokenize on reload
        self.index_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "chunks": self._id_to_chunk,
            "corpus_tokens": self._corpus_tokens,
        }
        path = self.index_dir / "bm25_meta.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=0)

    def _load_metadata(self) -> bool:
        path = self.index_dir / "bm25_meta.json"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._id_to_chunk = meta["chunks"]
        self._corpus_tokens = meta["corpus_tokens"]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        return True

    def load_index(self, chunk_metadatas: Optional[List[dict]] = None) -> bool:
        if chunk_metadatas is not None:
            self._id_to_chunk = chunk_metadatas
            return self._load_corpus_only()
        return self._load_metadata()

    def _load_corpus_only(self) -> bool:
        # load corpus tokens and rebuild BM25
        path = self.index_dir / "bm25_meta.json"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._corpus_tokens = meta["corpus_tokens"]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        return True

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[dict, float]]:
        if self._bm25 is None:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            if scores[i] <= 0:
                break
            results.append((self._id_to_chunk[i], float(scores[i])))
        return results
