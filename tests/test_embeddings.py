import pytest

from src import config
from src.embeddings import EmbeddingService

requires_live = pytest.mark.skipif(
    not (config.RUN_LIVE_TESTS and config.GEMINI_API_KEY),
    reason="Set RUN_LIVE_TESTS=true and GEMINI_API_KEY to run live Gemini API tests.",
)


def test_embedding_service_can_be_constructed_without_api_key():
    # Construction itself must not require network access or a key --
    # only the first actual embed call does.
    service = EmbeddingService(api_key="unused-placeholder")
    assert service.model


def test_embedding_service_raises_without_api_key():
    service = EmbeddingService(api_key="")
    with pytest.raises(RuntimeError):
        _ = service.client


@requires_live
def test_live_embedding_service_initializes():
    service = EmbeddingService()
    assert service.client is not None


@requires_live
def test_live_embed_documents_returns_vectors():
    service = EmbeddingService()
    vectors = service.embed_documents(["hello world", "second chunk"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[0]) == len(vectors[1])


@requires_live
def test_live_embed_query_returns_vector_with_matching_dimension():
    service = EmbeddingService()
    doc_vectors = service.embed_documents(["hello world"])
    query_vector = service.embed_query("hello")
    assert len(query_vector) == len(doc_vectors[0])
