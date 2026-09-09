"""
Splits loaded Documents into overlapping text chunks while preserving
`source` / `page` metadata, ready for embedding.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split documents into chunks, dropping any that end up empty."""
    chunk_size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    # Filter out empty/whitespace-only chunks (e.g. blank PDF pages).
    return [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
