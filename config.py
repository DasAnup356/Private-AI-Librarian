# config file - all paths and model settings in one place
import os
from pathlib import Path


def _detect_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# GPU config
USE_GPU = _detect_gpu()
LLM_GPU_LAYERS = -1 if USE_GPU else 0  # -1 = use all GPU layers, 0 = CPU only
DEVICE = "cuda" if USE_GPU else "cpu"

# paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_DIR = PROJECT_ROOT / "index"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# reranker
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM - put GGUF file in models/ folder
LLM_GGUF_PATH = MODELS_DIR / "llama-3-8b-instruct-q4.gguf"
LLM_GGUF_GLOB = "*.gguf"

# chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MIN_CHUNK_CHARS = 100

# retrieval settings
TOP_K_DENSE = 20
TOP_K_SPARSE = 20
TOP_K_AFTER_MERGE = 30
TOP_K_AFTER_RERANK = 8

# LLM generation settings
LLM_MAX_TOKENS = 1024
LLM_TEMP = 0.2
LLM_N_CTX = 4096

# index filenames
FAISS_INDEX_NAME = "faiss.index"
DOCSTORE_NAME = "docstore.json"
