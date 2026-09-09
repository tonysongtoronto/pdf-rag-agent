"""
Wraps a local FAISS index built via LangChain's FAISS vector store.

Responsible for: creating the index, adding chunks, saving it to disk,
and loading it back on a later run so PDFs never need to be re-processed
or re-embedded just to answer a query.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src import config
from src.embeddings import EmbeddingService


class VectorStore:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        persist_dir: Optional[Path] = None,
        index_name: Optional[str] = None,
    ) -> None:
        self.embedding_service = embedding_service or EmbeddingService()
        self.persist_dir = Path(persist_dir) if persist_dir else config.VECTORSTORE_DIR
        self.index_name = index_name or config.FAISS_INDEX_NAME
        self._store: FAISS | None = None

    # -- creation / persistence -------------------------------------------------
    def build_from_documents(self, documents: List[Document]) -> FAISS:
        """Create a brand new FAISS index from chunked documents."""
        if not documents:
            raise ValueError("Cannot build a FAISS index from zero documents.")
        self._store = FAISS.from_documents(documents, self.embedding_service.client)
        return self._store

    def add_documents(self, documents: List[Document]) -> None:
        """Add more chunks to an already-built/loaded index."""
        if self._store is None:
            self.build_from_documents(documents)
        else:
            self._store.add_documents(documents)

    def save(self) -> None:
        if self._store is None:
            raise RuntimeError("No FAISS index in memory to save.")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.persist_dir), index_name=self.index_name)

    def load(self) -> FAISS:
        """Load a previously saved index from disk."""
        self._store = FAISS.load_local(
            str(self.persist_dir),
            self.embedding_service.client,
            index_name=self.index_name,
            allow_dangerous_deserialization=True,
        )
        return self._store

    def exists_on_disk(self) -> bool:
        return (self.persist_dir / f"{self.index_name}.faiss").exists()

    # -- querying ---------------------------------------------------------------
    @property
    def store(self) -> FAISS:
        if self._store is None:
            if self.exists_on_disk():
                self.load()
            else:
                raise RuntimeError(
                    "No FAISS index available. Run `python ingest.py` first."
                )
        return self._store

    def similarity_search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        top_k = top_k if top_k is not None else config.TOP_K
        return self.store.similarity_search(query, k=top_k)
