"""
Integration tests for the /documents endpoints through the real
FastAPI app -- upload, list, ingest, and delete -- using a fake
embedding backend (via monkeypatch) so no GEMINI_API_KEY is needed and
no network calls happen. Agent() itself is real: it's cheap to
construct and only touches the FAISS index/DeepSeek when .ask() is
actually called, which none of these tests do.
"""
from __future__ import annotations

import shutil

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config


@pytest.fixture
def client(tmp_path, monkeypatch, fake_embedding_service):
    """A TestClient wired to isolated DATA_DIR/VECTORSTORE_DIR, with
    VectorStore's default EmbeddingService swapped for the fake one so
    ingestion never needs a real GEMINI_API_KEY."""
    data_dir = tmp_path / "data"
    vectorstore_dir = tmp_path / "vectorstore"
    data_dir.mkdir()
    vectorstore_dir.mkdir()

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "VECTORSTORE_DIR", vectorstore_dir)
    monkeypatch.setattr(
        "src.vector_store.EmbeddingService", lambda *a, **k: fake_embedding_service
    )

    from src.api.routes import router

    app = FastAPI()
    app.state.agent = None
    app.include_router(router)
    return TestClient(app)


def _upload(client, filename: str):
    # Reuse the repo's real sample.pdf as upload content for every
    # test file -- only the filename (and thus the `source` metadata
    # used for grouping) differs between "files".
    from src.config import PROJECT_ROOT

    sample_path = PROJECT_ROOT / "data" / "sample.pdf"
    with open(sample_path, "rb") as f:
        return client.post(
            "/documents/upload", files={"file": (filename, f, "application/pdf")}
        )


def test_upload_then_list_shows_not_indexed(client):
    r = _upload(client, "a.pdf")
    assert r.status_code == 200

    docs = client.get("/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "a.pdf"
    assert docs[0]["indexed"] is False
    assert docs[0]["pages"] is not None


def test_ingest_marks_uploaded_file_as_indexed(client):
    _upload(client, "a.pdf")
    r = client.post("/ingest")
    assert r.status_code == 200
    body = r.json()
    assert body["pdf_count"] == 1
    assert body["added"] > 0

    docs = client.get("/documents").json()["documents"]
    assert docs[0]["indexed"] is True


def test_ingest_twice_skips_unchanged_file(client):
    _upload(client, "a.pdf")
    client.post("/ingest")
    r2 = client.post("/ingest")
    body = r2.json()
    assert body["added"] == 0
    assert body["skipped"] > 0


def test_ingest_with_no_files_returns_400(client):
    r = client.post("/ingest")
    assert r.status_code == 400


def test_delete_removes_file_and_its_vectors_immediately(client):
    _upload(client, "a.pdf")
    _upload(client, "b.pdf")
    client.post("/ingest")

    r = client.delete("/documents/a.pdf")
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"

    docs = client.get("/documents").json()["documents"]
    assert [d["filename"] for d in docs] == ["b.pdf"]
    # a.pdf's vectors should be gone without a separate /ingest call.
    from src.vector_store import VectorStore

    store = VectorStore()
    assert store.is_indexed("a.pdf") is False
    assert store.is_indexed("b.pdf") is True


def test_deleting_last_file_empties_index_without_error(client):
    _upload(client, "a.pdf")
    client.post("/ingest")

    r = client.delete("/documents/a.pdf")
    assert r.status_code == 200

    health = client.get("/health").json()
    assert health["index_ready"] is False
    assert client.get("/documents").json()["documents"] == []


def test_delete_nonexistent_file_returns_404(client):
    r = client.delete("/documents/nope.pdf")
    assert r.status_code == 404


def test_delete_rejects_path_traversal(client):
    r = client.delete("/documents/..%2F..%2Fetc%2Fpasswd")
    assert r.status_code == 404
