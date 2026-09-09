"""
Loads PDF files from the data directory and returns LangChain Document
objects with `source` (filename) and `page` metadata preserved.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader

from src import config


def load_pdf(path: Path) -> List[Document]:
    """Load a single PDF file into one Document per page."""
    reader = PdfReader(str(path))
    documents: List[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_number},
            )
        )
    return documents


def load_pdfs_from_directory(directory: Path | None = None) -> List[Document]:
    """Load every .pdf file found directly inside `directory`.

    Returns an empty list if the directory has no PDFs -- callers should
    decide whether that is an error.
    """
    directory = Path(directory) if directory else config.DATA_DIR
    documents: List[Document] = []
    if not directory.exists():
        return documents

    for pdf_path in sorted(directory.glob("*.pdf")):
        documents.extend(load_pdf(pdf_path))
    return documents
