"""
Route handlers. Each one is a thin wrapper around existing business
logic (src.agent.graph.Agent, src.ingestion.run_ingestion) -- no
LangChain/LangGraph/FAISS/Gemini/DeepSeek code lives here.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src import config, document_store
from src.agent.graph import Agent
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    DocumentItem,
    DocumentsListResponse,
    HealthResponse,
    IngestResponse,
    UploadResponse,
)
from src.ingestion import IngestionError, run_ingestion, sync_index_with_disk

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
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    config.ensure_dirs()
    destination = Path(config.DATA_DIR) / file.filename
    content = await file.read()
    destination.write_bytes(content)

    # File is on disk but NOT indexed yet -- that only happens on the
    # next POST /ingest. register_upload() records it as un-indexed so
    # GET /documents reports it accurately in the meantime.
    document_store.register_upload(file.filename)

    return UploadResponse(filename=file.filename, status="uploaded")


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
    target = Path(config.DATA_DIR) / safe_name


    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"No such document: {safe_name}")

    target.unlink()
    document_store.remove(safe_name)

    # Sync immediately rather than waiting for the next POST /ingest:
    # since sync_index_with_disk() re-chunks every *remaining* PDF
    # (cheap -- no embedding calls for content already in the index)
    # and only cleanup="full" needs the up-to-date picture to know
    # this file's vectors should go, there's no real cost to doing it
    # now. This also means deleting the very last file is NOT an
    # error (unlike POST /ingest with zero PDFs) -- it just empties
    # the index.
    try:
        sync_index_with_disk()
    except IngestionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    _refresh_agent(request)
    
    return DeleteResponse(
        filename=safe_name,
        status="deleted",
        note="Removed from disk, and its chunks were removed from the FAISS index.",
    )
