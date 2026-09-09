from src.document_loader import load_pdf, load_pdfs_from_directory
from src import config


def test_load_pdf_returns_documents(sample_pdf_path):
    documents = load_pdf(sample_pdf_path)
    assert len(documents) > 0


def test_load_pdf_preserves_source_and_page(sample_pdf_path):
    documents = load_pdf(sample_pdf_path)
    for i, doc in enumerate(documents, start=1):
        assert doc.metadata["source"] == "sample.pdf"
        assert doc.metadata["page"] == i


def test_load_pdfs_from_directory(sample_pdf_path):
    documents = load_pdfs_from_directory(config.DATA_DIR)
    assert len(documents) > 0
    assert all("source" in doc.metadata for doc in documents)
    assert all("page" in doc.metadata for doc in documents)


def test_load_pdfs_from_empty_directory(tmp_path):
    documents = load_pdfs_from_directory(tmp_path)
    assert documents == []
