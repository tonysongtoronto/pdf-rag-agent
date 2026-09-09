"""
Pydantic schemas defining the API contract. Kept separate from route
handlers so the shapes are easy to scan/share (e.g. for generating a
TypeScript client from the OpenAPI schema FastAPI exposes at /docs).
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")


class SourceItem(BaseModel):
    source: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


class IngestResponse(BaseModel):
    status: str
    pdf_count: int
    chunk_count: int


class ErrorResponse(BaseModel):
    detail: str


class UploadResponse(BaseModel):
    filename: str
    status: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    gemini_key_configured: bool
    deepseek_key_configured: bool
