"""
Independent retriever: takes a user question, embeds it via the shared
EmbeddingService (through the vector store), runs FAISS similarity
search, and returns the top-K Documents with source/page/content intact.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from src import config
from src.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: Optional[VectorStore] = None, top_k: Optional[int] = None) -> None:
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k if top_k is not None else config.TOP_K

    def invoke(self, question: str, top_k: Optional[int] = None) -> List[Document]:
        k = top_k if top_k is not None else self.top_k
        return self.vector_store.similarity_search(question, top_k=k)
