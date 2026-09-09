from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402


class FakeEmbeddings(Embeddings):
    """Deterministic, dependency-free stand-in for Gemini embeddings.

    Hashes each text into a small fixed-size vector so similarity search
    behaves consistently across a test run without any network call.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


class FakeEmbeddingService:
    """Drop-in replacement for src.embeddings.EmbeddingService in tests."""

    def __init__(self) -> None:
        self.client = FakeEmbeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.client.embed_query(text)


@pytest.fixture
def sample_pdf_path() -> Path:
    return config.DATA_DIR / "sample.pdf"


@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(
            page_content="Acme Corp's revenue increased by eighteen percent this year, "
            "driven by strong sales of the GadgetPro line.",
            metadata={"source": "company_report.pdf", "page": 12},
        ),
        Document(
            page_content="GadgetPro Max includes a longer battery life and a two-year "
            "warranty covering all defects.",
            metadata={"source": "product_manual.pdf", "page": 3},
        ),
        Document(
            page_content="This policy document outlines the company's remote work "
            "guidelines for all employees.",
            metadata={"source": "policy.pdf", "page": 1},
        ),
    ]


@pytest.fixture
def fake_embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def fake_vector_store(tmp_path, sample_documents, fake_embedding_service):
    from src.vector_store import VectorStore

    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    store.build_from_documents(sample_documents)
    return store


@pytest.fixture
def fake_retriever(fake_vector_store):
    from src.retriever import Retriever

    return Retriever(vector_store=fake_vector_store, top_k=2)
