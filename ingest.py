"""
Ingestion CLI: PDF -> Loader -> Chunk -> Gemini Embedding -> FAISS -> Save.

Usage:
    python ingest.py
"""
from __future__ import annotations

import sys

from src.ingestion import IngestionError, run_ingestion


def main() -> None:
    try:
        result = run_ingestion()
    except IngestionError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"PDF files processed: {result.pdf_count}")
    print(f"Chunks in current set: {result.chunk_count}")
    print(
        f"Synced: {result.added} added, {result.updated} updated, "
        f"{result.skipped} unchanged (skipped), {result.deleted} removed."
    )


if __name__ == "__main__":
    main()
