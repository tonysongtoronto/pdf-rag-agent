"""
The actual ingestion pipeline logic (PDF -> Loader -> Chunk -> Gemini
Embedding -> FAISS -> Save), factored out of ingest.py so both the CLI
script and the HTTP API can call the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.document_loader import load_pdfs_from_directory
from src.text_splitter import split_documents
from src.vector_store import VectorStore


class IngestionError(RuntimeError):
    """Raised when ingestion cannot proceed (no PDFs, no chunks, no API key)."""


@dataclass
class IngestionResult:
    pdf_count: int
    chunk_count: int


def run_ingestion() -> IngestionResult:
    """Run the full ingestion pipeline and persist the FAISS index.

    Raises IngestionError with a human-readable message on any failure
    (no PDFs found, nothing to chunk, missing API key, etc.) instead of
    letting a raw exception/traceback escape to the caller.
    """
    config.ensure_dirs()

    pdf_paths = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise IngestionError(f"No PDF files found in {config.DATA_DIR}. Add some and re-run.")

    documents = load_pdfs_from_directory(config.DATA_DIR)
    chunks = split_documents(documents)
    if not chunks:
        raise IngestionError("No text could be extracted/chunked from the PDFs.")

    vector_store = VectorStore()
    try:
        vector_store.build_from_documents(chunks)
        vector_store.save()
    except RuntimeError as exc:
        raise IngestionError(str(exc)) from exc

    return IngestionResult(pdf_count=len(pdf_paths), chunk_count=len(chunks))
