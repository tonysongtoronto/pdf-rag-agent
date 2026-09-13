"""
Mock-based tests for the "existing / is_indexed" decision matrix inside
POST /documents/upload (src/api/routes.py):

    existing = document_store.get(filename)
    if existing is not None:
        vector_store = VectorStore()
        if vector_store.is_indexed(filename):
            raise HTTPException(409, ...)
        # else: fall through, treat as a retry

document_store.get and VectorStore are each monkeypatched independently
(instead of driving them into a given state via real uploads/ingests),
so all four combinations of {existing True/False} x {is_indexed
True/False} can be exercised directly and cheaply -- no FAISS index,
no embedding service, no GEMINI_API_KEY needed.

    1. existing=True,  is_indexed=True  -> 409, re-upload blocked
    2. existing=True,  is_indexed=False -> 200, treated as a retry
    3. existing=False, is_indexed=True  -> 200, is_indexed never consulted
    4. existing=False, is_indexed=False -> 200, is_indexed never consulted

For the real upload -> index -> delete lifecycle (nothing mocked except
the embedding backend), see tests/test_upload_lifecycle.py.
"""

#    uv run pytest tests/test_upload_conflict_matrix.py tests/test_upload_lifecycle.py -v


from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config

FILENAME = "a.pdf"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to an isolated DATA_DIR/VECTORSTORE_DIR so the
    upload's real side effects (writing the PDF, writing
    documents_metadata.json via document_store.register_upload) don't
    touch the repo's actual data/ or vectorstore/ folders.
    """
    data_dir = tmp_path / "data"
    vectorstore_dir = tmp_path / "vectorstore"
    data_dir.mkdir()
    vectorstore_dir.mkdir()

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "VECTORSTORE_DIR", vectorstore_dir)

    from src.api.routes import router

    app = FastAPI()
    app.state.agent = None
    app.include_router(router)
    return TestClient(app)


# Content doesn't need to be a real, parseable PDF here: these tests
# only exercise the existing/is_indexed branch, and
# document_store.register_upload()'s page-count lookup already
# swallows unreadable-PDF errors and just records pages=None. Using
# dummy bytes avoids depending on a data/sample.pdf fixture file that
# may not exist in every checkout.
_DUMMY_PDF_BYTES = b"%PDF-1.4\n%dummy content for upload tests\n%%EOF"


def _upload(client, filename: str = FILENAME):
    return client.post(
        "/documents/upload",
        files={"file": (filename, _DUMMY_PDF_BYTES, "application/pdf")},
    )


def _mock_existing(monkeypatch, *, exists: bool):
    """Stand in for document_store.get(filename): a non-None record
    when the file is already tracked, None for a brand-new upload."""
    record = (
        {"filename": FILENAME, "pages": 1, "uploaded_at": "2024-01-01T00:00:00+00:00"}
        if exists
        else None
    )
    monkeypatch.setattr(
        "src.api.routes.document_store.get", lambda filename: record
    )


def _mock_vector_store(monkeypatch, *, indexed: bool) -> MagicMock:
    """Stand in for `VectorStore()` as used inside routes.py, so the
    route never has to construct a real FAISS-backed VectorStore (no
    embedding service / network call needed just to check a flag)."""
    fake_instance = MagicMock()
    fake_instance.is_indexed.return_value = indexed
    monkeypatch.setattr("src.api.routes.VectorStore", lambda: fake_instance)
    return fake_instance


# 1. existing=True, is_indexed=True -> 409, re-upload blocked
def test_existing_and_indexed_returns_409(client, monkeypatch):
    _mock_existing(monkeypatch, exists=True)
    fake_vs = _mock_vector_store(monkeypatch, indexed=True)

    r = _upload(client)

    assert r.status_code == 409
    assert "already exists and is indexed" in r.json()["detail"]
    fake_vs.is_indexed.assert_called_once_with(FILENAME)


# 2. existing=True, is_indexed=False -> 200, treated as a retry
def test_existing_but_not_indexed_allows_reupload(client, monkeypatch):
    _mock_existing(monkeypatch, exists=True)
    fake_vs = _mock_vector_store(monkeypatch, indexed=False)

    r = _upload(client)

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "uploaded"
    assert body["indexed"] is False
    fake_vs.is_indexed.assert_called_once_with(FILENAME)


# 3. existing=False, is_indexed=True -> 200, is_indexed never consulted
# (the `if existing is not None` guard short-circuits before it's read)
def test_new_file_never_checks_is_indexed_even_if_true(client, monkeypatch):
    _mock_existing(monkeypatch, exists=False)
    fake_vs = _mock_vector_store(monkeypatch, indexed=True)

    r = _upload(client)

    assert r.status_code == 200
    assert r.json()["status"] == "uploaded"
    fake_vs.is_indexed.assert_not_called()


# 4. existing=False, is_indexed=False -> 200, is_indexed never consulted
def test_new_file_never_checks_is_indexed_when_false(client, monkeypatch):
    _mock_existing(monkeypatch, exists=False)
    fake_vs = _mock_vector_store(monkeypatch, indexed=False)

    r = _upload(client)

    assert r.status_code == 200
    assert r.json()["status"] == "uploaded"
    fake_vs.is_indexed.assert_not_called()