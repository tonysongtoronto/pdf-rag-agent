"""
Single, decoupled entry point for all embedding calls (Gemini).

No other module should instantiate a Gemini embedding client directly --
they should depend on `EmbeddingService` instead. This keeps the
provider swappable and makes it trivial to mock in tests.
"""
from __future__ import annotations

from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src import config


class EmbeddingService:
    """Wraps the Gemini embedding model behind a small, stable interface."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or config.EMBEDDING_MODEL
        # self._api_key = api_key or config.GEMINI_API_KEY
        self._api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        self._client: GoogleGenerativeAIEmbeddings | None = None

    @property
    def client(self) -> GoogleGenerativeAIEmbeddings:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Add it to your .env file before "
                    "embedding documents or queries."
                )
            self._client = GoogleGenerativeAIEmbeddings(
                model=self.model,
                google_api_key=self._api_key,
            )
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document chunks."""
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single user question."""
        return self.client.embed_query(text)
