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
    print(f"Chunks created: {result.chunk_count}")
    print("FAISS index created successfully.")


if __name__ == "__main__":
    main()
