"""
Route handlers. Each one is a thin wrapper around existing business
logic (src.agent.graph.Agent, src.ingestion.run_ingestion) -- no
LangChain/LangGraph/FAISS/Gemini/DeepSeek code lives here.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src import config, document_store
from src.agent.graph import Agent
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentItem,
    DocumentsListResponse,
    HealthResponse,
    IndexResponse,
    IngestResponse,
    UploadResponse,
)
from src.document_loader import load_pdf
from src.ingestion import IngestionError, run_ingestion, sync_index_with_disk
from src.text_splitter import split_documents
from src.vector_store import VectorStore

router = APIRouter()


def _get_agent(request: Request) -> Agent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="No FAISS index available yet. Call POST /ingest first.",
        )
    return agent


def _refresh_agent(request: Request) -> None:
    """Rebuild app.state.agent so /chat immediately reflects whatever
    the index looks like right now, instead of serving stale results
    from the previous one. Same try/except pattern as the startup
    lifespan in app.py: if the index is now empty (e.g. the last file
    was just deleted), Agent() raises RuntimeError -- that's a valid
    state, not a crash, so /chat should report a clean 503 rather
    than the server falling over.
    """
    try:
        request.app.state.agent = Agent()
    except RuntimeError:
        request.app.state.agent = None


def _index_one_file(filename: str) -> dict:
    """Load, chunk, and sync exactly one file into FAISS via
    VectorStore.sync_one() (cleanup="scoped_full") -- shared by
    POST /documents/upload?auto_index=true and
    POST /documents/{filename}/index so there's one code path for
    "index this one file" instead of two.
    """
    path = Path(config.DATA_DIR) / filename
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"'{filename}' is not on disk to index. Upload it again first "
            "(disk storage is temporary staging, not persisted long-term).",
        )

    documents = load_pdf(path)
    chunks = split_documents(documents)
    if not chunks:
        raise HTTPException(status_code=422, detail=f"No extractable text in '{filename}'.")

    vector_store = VectorStore()
    result = vector_store.sync_one(filename, chunks)

    return {
        "filename": filename,
        "pages": len(documents),
        "chunks_indexed": result["num_added"] + result["num_updated"],
        "status": "indexed",
    }


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    # Agent() construction is lazy (no FAISS/Gemini/DeepSeek calls happen
    # until a question actually runs), so "does the index exist on disk"
    # is a more honest signal than "did app.state.agent get set".
    vector_store_exists = (config.VECTORSTORE_DIR / f"{config.FAISS_INDEX_NAME}.faiss").exists()
    return HealthResponse(
        status="ok",
        index_ready=vector_store_exists,
        gemini_key_configured=bool(config.GEMINI_API_KEY),
        deepseek_key_configured=bool(config.DEEPSEEK_API_KEY),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    agent = _get_agent(request)
    try:
        result = agent.ask(payload.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(**result)


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: Request) -> IngestResponse:
    try:
        result = run_ingestion()
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _refresh_agent(request)

    return IngestResponse(
        status="success",
        pdf_count=result.pdf_count,
        chunk_count=result.chunk_count,
        added=result.added,
        updated=result.updated,
        skipped=result.skipped,
        deleted=result.deleted,
    )


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    auto_index: bool = Form(False),
    request: Request = None,
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    # Path(...).name strips any directory components a caller might
    # sneak into the filename (e.g. "../../etc/passwd").
    filename = Path(file.filename).name

    existing = document_store.get(filename)
    if existing is not None:
        vector_store = VectorStore()
        if vector_store.is_indexed(filename):
            raise HTTPException(
                status_code=409,
                detail=f"'{filename}' already exists and is indexed. "
                "Delete it before re-uploading.",
            )
        # existing but not indexed: a prior upload never made it into
        # FAISS (e.g. disk was wiped before /index ran, since disk is
        # only temporary staging). Nothing committed to protect, so
        # this re-upload is a retry, not a conflict.

    config.ensure_dirs()
    destination = Path(config.DATA_DIR) / filename
    content = await file.read()
    destination.write_bytes(content)

    # File is on disk but NOT indexed yet -- that only happens once
    # /index runs, whether triggered below (auto_index) or later via
    # POST /documents/{filename}/index. register_upload() records it
    # as un-indexed so GET /documents reports it accurately either way.
    document_store.register_upload(filename)

    if not auto_index:
        return UploadResponse(filename=filename, status="uploaded", indexed=False)

    index_result = _index_one_file(filename)
    if request is not None:
        _refresh_agent(request)
    return UploadResponse(
        filename=filename,
        status="uploaded",
        indexed=True,
        index_result=IndexResponse(**index_result),
    )


@router.post("/documents/{filename}/index", response_model=IndexResponse)
def index_document(filename: str, request: Request) -> IndexResponse:
    safe_name = Path(filename).name
    if document_store.get(safe_name) is None:
        raise HTTPException(status_code=404, detail=f"'{safe_name}' has not been uploaded yet.")

    result = _index_one_file(safe_name)
    _refresh_agent(request)
    return IndexResponse(**result)


@router.get("/documents", response_model=DocumentsListResponse)
def list_documents() -> DocumentsListResponse:
    documents = [DocumentItem(**item) for item in document_store.list_documents()]
    return DocumentsListResponse(documents=documents)


@router.delete("/documents/{filename}", response_model=DeleteResponse)
def delete_document(filename: str, request: Request) -> DeleteResponse:
    # Path(...).name strips any directory components a caller might
    # sneak into the URL segment (e.g. "../../etc/passwd") so this can
    # only ever touch a file directly inside DATA_DIR.
    safe_name = Path(filename).name

    if document_store.get(safe_name) is None:
        raise HTTPException(status_code=404, detail=f"No such document: {safe_name}")

    # Targeted, per-file removal (scoped to this file's own record
    # manager group) instead of a full-corpus resync -- other files'
    # chunks are never touched.
    vector_store = VectorStore()
    removed = vector_store.delete_source(safe_name)

    # Disk is temporary staging, not the source of truth: delete the
    # file if it's there, but a missing file is not an error -- it may
    # already be gone (storage that doesn't persist across restarts).
    note = None
    target = Path(config.DATA_DIR) / safe_name
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        note = f"Vector(s) and record removed, but could not delete the file on disk: {exc}"

    document_store.remove(safe_name)
    _refresh_agent(request)

    return DeleteResponse(
        filename=safe_name,
        status="deleted",
        note=note or f"Removed {removed} chunk(s) from the FAISS index.",
    )