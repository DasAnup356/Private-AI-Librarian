# load docs, chunk, embed, build faiss + bm25
from pathlib import Path
from typing import List, Optional

from config import INDEX_DIR, UPLOADS_DIR
from ingestion import load_document, load_documents_from_directory
from chunking import chunk_by_headings_and_sliding, Chunk
from chunking.strategies import chunks_to_texts
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from retrieval.docstore import save_docstore


def build_metadata(chunk: Chunk, global_index: int) -> dict:
    return {
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "heading": chunk.heading,
        "global_index": global_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def run_indexing(
    source: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> dict:
    index_dir = Path(index_dir or INDEX_DIR)
    source = Path(source or UPLOADS_DIR)

    if source.is_file():
        pairs = []
        text = load_document(source)
        if text:
            pairs.append((source, text))
    else:
        pairs = load_documents_from_directory(source)

    if not pairs:
        return {"n_docs": 0, "n_chunks": 0, "status": "no_documents"}

    all_chunks: List[Chunk] = []
    for path, text in pairs:
        doc_id = path.stem
        chunks = chunk_by_headings_and_sliding(text, doc_id)
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"n_docs": len(pairs), "n_chunks": 0, "status": "no_chunks"}

    chunk_texts = chunks_to_texts(all_chunks)
    chunk_metadatas = [build_metadata(c, i) for i, c in enumerate(all_chunks)]

    save_docstore(chunk_texts, chunk_metadatas, base_dir=index_dir)

    dense = DenseRetriever(index_dir=index_dir)
    dense.build_index(chunk_texts, chunk_metadatas)

    sparse = SparseRetriever(index_dir=index_dir)
    sparse.build_index(chunk_texts, chunk_metadatas)

    return {
        "n_docs": len(pairs),
        "n_chunks": len(all_chunks),
        "status": "ok",
    }
