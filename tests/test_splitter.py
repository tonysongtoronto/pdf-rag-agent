from langchain_core.documents import Document

from src.text_splitter import split_documents


def test_split_documents_produces_chunks(sample_documents):
    chunks = split_documents(sample_documents, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 0


def test_chunks_are_not_empty(sample_documents):
    chunks = split_documents(sample_documents, chunk_size=50, chunk_overlap=10)
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_chunks_preserve_metadata(sample_documents):
    chunks = split_documents(sample_documents, chunk_size=50, chunk_overlap=10)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata


def test_blank_documents_produce_no_chunks():
    blank = [Document(page_content="   ", metadata={"source": "x.pdf", "page": 1})]
    chunks = split_documents(blank)
    assert chunks == []
