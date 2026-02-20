# chunk texts + metadata in json
import json
from pathlib import Path
from typing import List, Optional

from config import INDEX_DIR


def docstore_path(base_dir: Optional[Path] = None) -> Path:
    return Path(base_dir or INDEX_DIR) / "docstore.json"


def save_docstore(
    chunk_texts: List[str],
    # save chunks and metadata to JSON
    # save chunks and metadata to JSON
    chunk_metadatas: List[dict],
    base_dir: Optional[Path] = None,
) -> None:
    path = docstore_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": chunk_texts,
        "metadatas": chunk_metadatas,
    }
    with open(path, "w", encoding="utf-8") as f:
    # load docstore from disk, returns (texts, metadatas) or empty lists
        json.dump(payload, f, ensure_ascii=False, indent=0)
    # load docstore from disk, returns (texts, metadatas) or empty lists


def load_docstore(base_dir: Optional[Path] = None) -> tuple:
    path = docstore_path(base_dir)
    if not path.exists():
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("chunks", []), payload.get("metadatas", [])
