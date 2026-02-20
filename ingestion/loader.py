# load pdf / txt
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader


def load_document(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_txt(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    return None


def _load_txt(path: Path) -> str:
    # try different encodings
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> Optional[str]:
    # extract text from PDF (text-based only, no OCR)
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts) if parts else None
    except Exception:
        return None


def load_documents_from_directory(
    directory: Path, extensions: tuple = (".pdf", ".txt")
) -> List[tuple]:
    directory = Path(directory)
    if not directory.is_dir():
        return []

    results = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in extensions:
            text = load_document(path)
            if text and text.strip():
                results.append((path, text))
    return results
