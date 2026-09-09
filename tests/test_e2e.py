import pytest

from src import config
from src.agent.graph import Agent
from src.document_loader import load_pdfs_from_directory
from src.text_splitter import split_documents
from src.vector_store import VectorStore
from src.retriever import Retriever

requires_live = pytest.mark.skipif(
    not (config.RUN_LIVE_TESTS and config.GEMINI_API_KEY and config.DEEPSEEK_API_KEY),
    reason="Set RUN_LIVE_TESTS=true, GEMINI_API_KEY and DEEPSEEK_API_KEY for the full live E2E test.",
)


def test_pipeline_builds_index_with_fakes(fake_embedding_service, tmp_path):
    """Runs the full ingestion pipeline (loader -> chunk -> embed -> FAISS ->
    retriever) against the real sample PDF, using a fake embedding backend
    so this test needs no API keys."""
    documents = load_pdfs_from_directory(config.DATA_DIR)
    chunks = split_documents(documents)
    assert len(chunks) > 0

    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    store.build_from_documents(chunks)
    store.save()

    retriever = Retriever(vector_store=store, top_k=3)
    results = retriever.invoke("What is the company's revenue?")
    assert len(results) > 0


@requires_live
def test_full_live_pipeline_answers_question():
    """True end-to-end test against the real Gemini + DeepSeek APIs."""
    documents = load_pdfs_from_directory(config.DATA_DIR)
    chunks = split_documents(documents)

    store = VectorStore()
    store.build_from_documents(chunks)
    retriever = Retriever(vector_store=store)

    agent = Agent(retriever=retriever)
    response = agent.ask("What is the main purpose of this document?")

    assert response["answer"]
    assert len(response["sources"]) > 0
