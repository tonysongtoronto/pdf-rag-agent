"""
The ingestion pipeline: PDF -> Loader -> Chunk -> sync into FAISS via
LangChain's Indexing API (src.vector_store.VectorStore.sync_documents),
factored out of ingest.py so the CLI script, POST /ingest, and
DELETE /documents/{filename} (which needs to sync too, to actually
remove a deleted file's vectors) all share one code path.

Every call re-reads and re-chunks every PDF currently in DATA_DIR --
that part is cheap (no API calls). Only chunks whose content hash
isn't already recorded get embedded; chunks for files no longer
present get deleted. See src/vector_store.py for the mechanics.
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
    added: int = 0
    updated: int = 0
    skipped: int = 0
    deleted: int = 0


def sync_index_with_disk() -> IngestionResult:
    """Reconcile the FAISS index with whatever PDFs are in DATA_DIR
    right now -- including zero PDFs, which legitimately means "delete
    everything that's left" (e.g. after DELETE /documents/{filename}
    removed the last file).

    Does NOT raise just because DATA_DIR is empty -- that's a valid
    post-deletion state, not a failure. DOES raise if PDFs exist but
    nothing could be extracted from any of them, since that smells
    like a transient read failure (corrupt file, permissions) rather
    than a deliberate "there's nothing here" -- syncing an empty chunk
    list in that case would wipe out a perfectly good index.
    """
    config.ensure_dirs()

    pdf_paths = sorted(config.DATA_DIR.glob("*.pdf"))
    documents = load_pdfs_from_directory(config.DATA_DIR) if pdf_paths else []
    chunks = split_documents(documents) if documents else []

    if pdf_paths and not chunks:
        raise IngestionError("No text could be extracted/chunked from the PDFs.")

    vector_store = VectorStore()
    try:
        sync_result = vector_store.sync_documents(chunks)
    except RuntimeError as exc:
        raise IngestionError(str(exc)) from exc

    return IngestionResult(
        pdf_count=len(pdf_paths),
        chunk_count=len(chunks),
        added=sync_result["num_added"],
        updated=sync_result["num_updated"],
        skipped=sync_result["num_skipped"],
        deleted=sync_result["num_deleted"],
    )


def run_ingestion() -> IngestionResult:
    """Used by POST /ingest. Same reconciliation as
    sync_index_with_disk(), but treats "no PDFs in DATA_DIR at all" as
    an error: a caller who explicitly asked to ingest and has nothing
    to ingest almost certainly forgot to upload a file first, and
    should be told so instead of getting a silent no-op 200. (DELETE
    calls sync_index_with_disk() directly, so deleting the very last
    file is NOT treated as an error there.)
    """
    result = sync_index_with_disk()
    if result.pdf_count == 0:
        raise IngestionError(f"No PDF files found in {config.DATA_DIR}. Add some and re-run.")
    return result
