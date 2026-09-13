"""
Wraps a local FAISS index built via LangChain's FAISS vector store,
kept in sync with what's actually on disk via LangChain's Indexing
API -- a RecordManager backed by a local sqlite file that remembers
a content hash per chunk.

Why a RecordManager instead of hand-rolling "which chunks came from
which file": calling sync_documents() with the FULL current set of
chunks lets the RecordManager figure out, on its own, what's new
(embed it), what's unchanged (skip it -- no embedding call), and
what's missing compared to last time (delete it from FAISS) --
without this code ever needing to track a FAISS-internal chunk id
itself. See: https://python.langchain.com/docs/how_to/indexing/
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from langchain_classic.indexes import SQLRecordManager
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.indexing import index

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
        self._record_manager: SQLRecordManager | None = None

    # -- indexing API record manager --------------------------------------------
    @property
    def record_manager(self) -> SQLRecordManager:
        """Lazily created, backed by a sqlite file next to the FAISS
        index files (same persist_dir) so tests using a tmp_path get a
        fully isolated record manager for free, same as they already
        get an isolated FAISS index."""
        if self._record_manager is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.persist_dir / config.RECORD_MANAGER_DB_NAME
            self._record_manager = SQLRecordManager(
                namespace=self.index_name,
                db_url=f"sqlite:///{db_path}",
            )
            self._record_manager.create_schema()
        return self._record_manager

    # -- creation / persistence -------------------------------------------------
    def build_from_documents(self, documents: List[Document]) -> FAISS:
        """Create a brand new FAISS index from chunked documents, with
        no Indexing API bookkeeping. Kept for callers that just want a
        one-shot index (tests, the `Retriever` fixtures); prefer
        sync_documents() for anything that gets re-run against a
        changing DATA_DIR.
        """
        if not documents:
            raise ValueError("Cannot build a FAISS index from zero documents.")
        self._store = FAISS.from_documents(documents, self.embedding_service.client)
        return self._store

    def add_documents(self, documents: List[Document]) -> None:
        """Add more chunks to an already-built/loaded index, with no
        dedup/deletion bookkeeping. See sync_documents() for the
        tracked version used by the actual ingestion pipeline."""
        if self._store is None:
            self.build_from_documents(documents)
        else:
            self._store.add_documents(documents)

    def _run_index(self, documents: List[Document], cleanup: str) -> dict:
        """Shared plumbing behind sync_documents()/sync_one(): make sure
        a store exists, call LangChain's index(), then persist-or-forget
        depending on whether anything is left. `cleanup` is "full" for
        the whole-corpus path or "scoped_full" for the single-file path
        -- see sync_documents() and sync_one() for which to use when.
        """
        if self._store is None:
            if self.exists_on_disk():
                self.load()
            else:
                self._bootstrap_empty_store()

        result = index(
            documents,
            self.record_manager,
            self._store,
            cleanup=cleanup,
            source_id_key="source",
            # sha1 is the library default but triggers a deprecation
            # warning on every call; blake2b is the recommended,
            # equally-fast replacement.
            key_encoder="blake2b",
        )

        if len(self._store.index_to_docstore_id) == 0:
            # Don't rely on FAISS's save/load round-trip being well
            # behaved for a genuinely empty index -- that's a path
            # almost nothing exercises. Treat "synced down to nothing"
            # the same as "never built": wipe the persisted files and
            # drop the in-memory store, so /health and `.store` see one
            # consistent "no index" state either way, not two.
            self._forget()
        else:
            self.save()

        return dict(result)

    def sync_documents(self, documents: List[Document]) -> dict:
        """Reconcile FAISS with `documents` via LangChain's Indexing
        API (cleanup="full", grouped by the `source` metadata key set
        by the document loader).

        Pass the FULL current set of chunks -- every file currently in
        DATA_DIR, not just new ones. Re-loading/re-chunking unchanged
        PDFs is cheap (no embedding calls); cleanup="full" needs to see
        everything that SHOULD exist right now to know what no longer
        does and delete it. Returns the
        {"num_added", "num_updated", "num_skipped", "num_deleted"}
        result from LangChain's index(). Kept for the admin-only
        POST /ingest reconciliation path; per-file uploads should use
        sync_one() instead (see below).
        """
        return self._run_index(documents, cleanup="full")

    def sync_one(self, filename: str, documents: List[Document]) -> dict:
        """(Re)index exactly one file's chunks without touching any
        other file's vectors.

        Pass ONLY this file's current chunks (every Document must have
        metadata["source"] == filename). Uses cleanup="scoped_full",
        which -- unlike "full" -- only looks at source ids it sees in
        this call: it adds/updates what changed and deletes this file's
        own now-stale chunks, but leaves every other file's chunks
        alone. This is what makes a single-file "Index now" possible
        without re-scanning or re-touching the rest of the corpus.
        """
        return self._run_index(documents, cleanup="scoped_full")

    def delete_source(self, filename: str) -> int:
        """Remove every chunk belonging to `filename` from FAISS and
        the record manager, without touching any other file's chunks.
        Returns the number of chunks removed (0 if the file had none,
        e.g. it was uploaded but never indexed).
        """
        if not self.exists_on_disk():
            return 0
        if self._store is None:
            self.load()

        keys = self.record_manager.list_keys(group_ids=[filename])
        if not keys:
            return 0

        # The Indexing API uses each record manager key as the matching
        # FAISS docstore id (see langchain_core.indexing.api.index),
        # so these keys can be passed straight to FAISS's delete().
        self._store.delete(keys)
        self.record_manager.delete_keys(keys)

        if len(self._store.index_to_docstore_id) == 0:
            self._forget()
        else:
            self.save()

        return len(keys)

    def is_indexed(self, filename: str) -> bool:
        """Whether `filename` currently has any chunks in the index,
        per the record manager -- not per any bookkeeping of our own."""
        if not self.exists_on_disk():
            return False
        try:
            return bool(self.record_manager.list_keys(group_ids=[filename]))
        except Exception:
            return False

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

    def _bootstrap_empty_store(self) -> None:
        """FAISS.from_documents() needs at least one document to infer
        the embedding dimension -- build with one throwaway doc, then
        delete it immediately. Standard workaround (LangChain's FAISS
        wrapper has no from_texts([]) path), not a hack specific to
        this project."""
        placeholder = [
            Document(page_content="__placeholder__", metadata={"source": "__init__"})
        ]
        self._store = FAISS.from_documents(placeholder, self.embedding_service.client)
        self._store.delete([self._store.index_to_docstore_id[0]])

    def _forget(self) -> None:
        """Drop the in-memory store and delete persisted FAISS files,
        so an index that's been emptied out looks identical to one
        that was never built."""
        self._store = None
        for suffix in (".faiss", ".pkl"):
            (self.persist_dir / f"{self.index_name}{suffix}").unlink(missing_ok=True)

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