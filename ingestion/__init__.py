"""
Document ingestion: load PDF and TXT files into raw text for chunking.
"""
from .loader import load_document, load_documents_from_directory

__all__ = ["load_document", "load_documents_from_directory"]
