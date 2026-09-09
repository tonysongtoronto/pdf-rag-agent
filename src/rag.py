"""
RAG orchestration: formats retrieved documents into a citation-friendly
string block, used by the search_documents tool.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document


def format_documents_for_context(documents: List[Document]) -> str:
    """Render retrieved documents as `Document: x, Page: y\n\ncontent` blocks."""
    if not documents:
        return "No relevant documents were found in the knowledge base."

    blocks = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        blocks.append(f"Document: {source}\nPage: {page}\n\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def documents_to_sources(documents: List[Document]) -> List[dict]:
    """Deduplicated list of {source, page} dicts, in first-seen order."""
    seen = set()
    sources = []
    for doc in documents:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": key[0], "page": key[1]})
    return sources
