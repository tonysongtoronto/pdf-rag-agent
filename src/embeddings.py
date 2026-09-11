"""
Single, decoupled entry point for all embedding calls (Gemini).

No other module should instantiate a Gemini embedding client directly --
they should depend on `EmbeddingService` instead. This keeps the
provider swappable and makes it trivial to mock in tests.

Wrapped in a local, persistent cache (LangChain's CacheBackedEmbeddings)
so re-embedding identical text -- a chunk re-synced after the
RecordManager resets, or the same test question asked again -- is
served from disk instead of spending another call against Gemini's
tight free-tier rate limit.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src import config

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True for Gemini's 429/RESOURCE_EXHAUSTED quota error and for
    503/UNAVAILABLE transient outages -- both are worth retrying.
    Any other GoogleGenerativeAIError (bad key, bad request, etc.)
    should fail immediately instead of being retried."""
    if not isinstance(exc, GoogleGenerativeAIError):
        return False
    message = str(exc)
    return any(
        marker in message
        for marker in ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE")
    )

_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class EmbeddingService:
    """Wraps the Gemini embedding model behind a small, stable interface."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or config.EMBEDDING_MODEL
        self._api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        self._client: CacheBackedEmbeddings | None = None

    @property
    def client(self) -> CacheBackedEmbeddings:
        """Gemini embeddings wrapped in a local on-disk cache, keyed by
        model name so switching EMBEDDING_MODEL never returns another
        model's stale vectors. Caches BOTH document chunks and user
        queries -- re-embedding identical text is served from disk
        with zero Gemini calls."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to your .env file before "
                    "embedding documents or queries."
                )
            underlying = GoogleGenerativeAIEmbeddings(
                model=self.model,
                google_api_key=self._api_key,
            )
            store = LocalFileStore(str(config.EMBEDDING_CACHE_DIR))
            self._client = CacheBackedEmbeddings.from_bytes_store(
                underlying,
                store,
                namespace=self.model,
                query_embedding_cache=True,
            )
        return self._client

    @_retry_on_rate_limit
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks. Cache hits skip Gemini
        entirely; cache misses retry with exponential backoff on
        Gemini's free-tier 429 quota error."""
        return self.client.embed_documents(texts)

    @_retry_on_rate_limit
    def embed_query(self, text: str) -> List[float]:
        """Embed a single user question. Same caching/retry behavior
        as embed_documents()."""
        return self.client.embed_query(text)