"""
FastAPI application entry point.

Run with:
    uv run uvicorn src.api.app:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.graph import Agent
from src.api.routes import router

from fastapi.responses import RedirectResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the Agent (and load the FAISS index) once at startup rather
    # than per-request. If no index exists yet (ingest.py / POST /ingest
    # hasn't run), leave it as None -- /chat reports a clear 503 instead
    # of crashing the server.
    try:
        app.state.agent = Agent()
    except RuntimeError:
        app.state.agent = None
    yield


app = FastAPI(title="PDF RAG Agent API", version="1.0.0", lifespan=lifespan)

# Adjust allow_origins for your actual frontend origin(s) in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# uv run uvicorn src.api.app:app --reload --port 8000

# 浏览器打开 http://127.0.0.1:8000/docs 看交互式文档。
