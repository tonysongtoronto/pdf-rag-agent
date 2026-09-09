"""
Route handlers. Each one is a thin wrapper around existing business
logic (src.agent.graph.Agent, src.ingestion.run_ingestion) -- no
LangChain/LangGraph/FAISS/Gemini/DeepSeek code lives here.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src import config
from src.agent.graph import Agent
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    UploadResponse,
)
from src.ingestion import IngestionError, run_ingestion

router = APIRouter()


def _get_agent(request: Request) -> Agent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="No FAISS index available yet. Call POST /ingest first.",
        )
    return agent


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

    # Rebuild the Agent so /chat immediately reflects the new index,
    # instead of serving stale results from the old one.
    request.app.state.agent = Agent()

    return IngestResponse(
        status="success",
        pdf_count=result.pdf_count,
        chunk_count=result.chunk_count,
    )


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    config.ensure_dirs()
    destination = Path(config.DATA_DIR) / file.filename
    content = await file.read()
    destination.write_bytes(content)

    return UploadResponse(filename=file.filename, status="uploaded")
