"""
Full lifecycle: upload -> index -> delete, through the real VectorStore /
document_store / FAISS record manager. Nothing is mocked except the
embedding backend (swapped for the deterministic FakeEmbeddingService
from tests/conftest.py, same as tests/test_routes_documents.py), so no
GEMINI_API_KEY or network call is needed.

For the mock-based unit tests covering the existing/is_indexed branch
in isolation, see tests/test_upload_conflict_matrix.py.
"""
from __future__ import annotations

import shutil

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import config


@pytest.fixture
def full_client(tmp_path, monkeypatch, fake_embedding_service):
    """Client wired to an isolated DATA_DIR/VECTORSTORE_DIR with a real
    (network-free) VectorStore, for exercising the actual
    upload -> index -> delete lifecycle end to end. VectorStore's
    default EmbeddingService is swapped for the deterministic fake one
    from conftest.py, same as tests/test_routes_documents.py.
    """
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

    yield TestClient(app)

    # Belt-and-suspenders cleanup. tmp_path is already an isolated,
    # pytest-managed temp directory -- this test never touches the
    # repo's real data/ or vectorstore/ folders regardless -- but this
    # removes the FAISS index / sqlite record-manager files the test
    # just wrote immediately, rather than waiting for pytest's own
    # (delayed, capped-at-N-runs) tmp_path garbage collection.
    shutil.rmtree(tmp_path, ignore_errors=True)


def _build_real_pdf_bytes(text: str = "Hello World") -> bytes:
    """A minimal, hand-built single-page PDF with real extractable
    text, so the real load_pdf() -> split_documents() ->
    VectorStore.sync_one() pipeline has something to actually chunk
    and embed."""
    content = f"BT /F1 24 Tf 10 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>"
        b"/MediaBox[0 0 200 200]/Contents 5 0 R>>",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]

    parts = [b"%PDF-1.4\n"]
    for i, body in enumerate(objects, start=1):
        parts.append(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    parts.append(
        f"5 0 obj\n<</Length {len(content)}>>\nstream\n".encode()
        + content
        + b"\nendstream\nendobj\n"
    )

    full_body = b"".join(parts)
    offsets = []
    pos = len(parts[0])
    for p in parts[1:]:
        offsets.append(pos)
        pos += len(p)

    xref_start = len(full_body)
    xref = "xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n"
    trailer = f"trailer\n<</Size 6/Root 1 0 R>>\nstartxref\n{xref_start}\n%%EOF"

    return full_body + xref.encode() + trailer.encode()


def test_full_upload_index_delete_lifecycle(full_client):
    """End to end: upload with auto_index=True indexes it immediately;
    it then shows up as indexed via GET /documents and /health; DELETE
    removes it from disk, FAISS, and document_store together and
    leaves the index in a clean, empty state -- no leftover file, no
    leftover vectors, no crash on the now-empty index."""
    client = full_client
    pdf_bytes = _build_real_pdf_bytes()

    # 1) Upload + index in one call.
    r = client.post(
        "/documents/upload",
        files={"file": ("lifecycle.pdf", pdf_bytes, "application/pdf")},
        data={"auto_index": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "uploaded"
    assert body["indexed"] is True
    assert body["index_result"]["chunks_indexed"] >= 1

    # 2) GET /documents confirms it shows up as indexed.
    docs = client.get("/documents").json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "lifecycle.pdf"
    assert docs[0]["indexed"] is True

    # 3) /health reports the index as ready.
    assert client.get("/health").json()["index_ready"] is True

    # 4) DELETE removes it from document_store, FAISS, and disk together.
    r = client.delete("/documents/lifecycle.pdf")
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"

    assert client.get("/documents").json()["documents"] == []

    from src.vector_store import VectorStore

    assert VectorStore().is_indexed("lifecycle.pdf") is False
    assert not (config.DATA_DIR / "lifecycle.pdf").exists()

    # 5) Deleting the only file empties the index cleanly.
    assert client.get("/health").json()["index_ready"] is False