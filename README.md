# Private AI Librarian

Offline RAG system for querying documents with a local LLM. Everything runs on your machine, no cloud APIs needed.

## What it does

- **Document ingestion**: Loads PDF and text files, extracts text content
- **Intelligent chunking**: Splits documents using heading-aware segmentation and sliding window with overlap
- **Dual indexing**: Builds both dense (semantic) and sparse (keyword) search indexes
  - Dense: FAISS index with BGE-small embeddings (384-dim vectors)
  - Sparse: BM25 inverted index for keyword matching
- **Hybrid retrieval**: Combines dense and sparse results using Reciprocal Rank Fusion (RRF)
- **Reranking**: Uses cross-encoder model to rerank retrieved chunks by relevance
- **Query rewriting**: LLM-based query expansion to improve retrieval quality
- **Answer generation**: Local LLM (GGUF) generates answers grounded in retrieved context
- **Confidence scoring**: Computes answer-context similarity as faithfulness metric

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download a GGUF model (like Llama 3 8B or Mistral 7B) and put it in the `models/` folder

3. Index your documents:
```bash
python main.py index
```

Or copy files to `data/uploads/` first, then run:
```bash
python main.py index --source data/uploads
```

4. Run the chat UI:
```bash
python main.py chat
```

Or query from command line:
```bash
python main.py query "your question"
```

## GPU Support

If you have a GPU, it'll use it automatically. Otherwise runs on CPU (slower but works fine).

Check GPU status: `python main.py gpu-check`

## How it works

**Indexing pipeline:**
1. Documents are parsed (PDF via pypdf, TXT with encoding fallback)
2. Text is chunked using heading detection (markdown headers, numbered sections, ALL CAPS) followed by sliding window chunking (512 tokens, 64 overlap)
3. Chunks are embedded using BGE-small-en-v1.5 (sentence-transformers) and stored in FAISS IndexFlatIP
4. Tokenized chunks are indexed with BM25 (rank_bm25) for sparse retrieval
5. Metadata (doc_id, chunk_index, headings, char offsets) is stored in JSON docstore

**Query pipeline:**
1. User query is rewritten by local LLM to expand/refine search terms
2. Dense retrieval: Query embedding → FAISS cosine similarity search (top-k=20)
3. Sparse retrieval: BM25 keyword matching (top-k=20)
4. Hybrid merge: RRF combines ranked lists (k=60) into unified ranking
5. Reranking: Cross-encoder (ms-marco-MiniLM-L-6-v2) scores query-chunk pairs, selects top-8
6. Context assembly: Selected chunks concatenated with separators (max 6000 chars)
7. Answer generation: Local LLM (GGUF) generates response using retrieved context
8. Confidence: Max cosine similarity between answer embedding and context chunk embeddings (0-1 scale)

## Project structure

```
models/          # put GGUF files here
data/uploads/    # upload PDFs/TXT here
index/           # generated indexes
ingestion/       # document loading
chunking/        # chunking
retrieval/       # dense, sparse, hybrid, reranker
llm/             # LLM stuff
evaluation/      # confidence scoring
ui/              # streamlit app
```

## Notes

- Works on CPU but GPU is much faster
- Needs ~8GB RAM minimum, 16GB recommended
- Uses BGE-small for embeddings, FAISS for dense search, BM25 for keyword search
