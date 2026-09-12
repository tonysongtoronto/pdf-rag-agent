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
import re
import threading
import time
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


_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")


def _wait_for_server_suggested_delay(retry_state) -> float:
    """Gemini's 429 body includes a RetryInfo.retryDelay (e.g. "41s") that
    tells you exactly how long is left in the current quota window --
    that's a much better signal than a blind exponential guess, which
    can easily under-wait and burn another attempt on the same 60s
    window. Fall back to exponential backoff if it isn't present (e.g.
    the transient 503 case, which has no such field)."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None:
        match = _RETRY_DELAY_RE.search(str(exc))
        if match:
            # Small buffer on top of Google's own number so we don't
            # race the exact millisecond the window resets.
            return float(match.group(1)) + 1.0
    return wait_exponential(multiplier=2, min=2, max=65)(retry_state)


_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit_error),
    wait=_wait_for_server_suggested_delay,
    # Free-tier RPM resets every 60s; 4 attempts at ~45-65s apart covers
    # several reset windows without hammering the API pointlessly if
    # something is fundamentally wrong (e.g. daily RPD quota exhausted,
    # which no amount of waiting-within-a-minute will fix).
    stop=stop_after_attempt(4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class _RateLimiter:
    """Proactive client-side throttle so we stop causing 429s instead of
    just reacting to them after the fact. Spaces out calls so we stay
    under `requests_per_minute` -- comfortably below Gemini's free-tier
    cap, not right at the edge, since other processes (pytest, a second
    ingest, concurrent FastAPI requests) may share the same quota."""

    def __init__(self, requests_per_minute: int) -> None:
        self._min_interval = 60.0 / max(requests_per_minute, 1)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            sleep_for = self._last_call + self._min_interval - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


# Shared across every EmbeddingService instance in this process, since
# the quota itself is shared per API key -- a second VectorStore/service
# built elsewhere in the same run must not reset the pacing clock.
_rate_limiter = _RateLimiter(config.EMBEDDING_RPM_LIMIT)

# Google's own hard cap on texts per embed_content request. We re-batch
# to this size ourselves (instead of handing embed_documents one huge
# list) so every individual Gemini request goes through the rate
# limiter -- letting the library's internal loop fire off several
# requests back-to-back is exactly what blows through 100 RPM in a
# single ingest.
_MAX_TEXTS_PER_REQUEST = 100


def _chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks. Cache hits skip Gemini
        entirely; cache misses are paced to stay under the free-tier
        RPM cap and retried (honoring Google's suggested retry delay)
        if a 429 slips through anyway.

        CacheBackedEmbeddings.embed_documents() checks the cache for
        the whole `texts` list before deciding what's actually missing,
        so we can't pre-split against the cache ourselves -- but we CAN
        cap how large a single call to it is, which keeps the number of
        underlying Gemini requests (and therefore rate-limiter waits)
        bounded even for a large PDF ingested in one go."""
        embeddings: List[List[float]] = []
        for batch in _chunked(texts, _MAX_TEXTS_PER_REQUEST):
            embeddings.extend(self._embed_documents_batch(batch))
        return embeddings

    @_retry_on_rate_limit
    def _embed_documents_batch(self, texts: List[str]) -> List[List[float]]:
        _rate_limiter.wait()
        return self.client.embed_documents(texts)

    @_retry_on_rate_limit
    def embed_query(self, text: str) -> List[float]:
        """Embed a single user question. Same pacing/retry behavior as
        embed_documents()."""
        _rate_limiter.wait()
        return self.client.embed_query(text)