"""
Central configuration for the PDF RAG + Agent backend.

All tunable parameters live here. Nothing else in the codebase should
read `os.environ` directly for these values -- import from this module
instead so there is a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
load_dotenv()

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore"))
EMBEDDING_CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", VECTORSTORE_DIR / "embedding_cache"))
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "faiss_index")
RECORD_MANAGER_DB_NAME = os.getenv("RECORD_MANAGER_DB_NAME", "record_manager.sqlite")

# --- API keys ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# --- Embedding -----------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

# Client-side pacing cap for embedding requests, kept comfortably below
# Gemini's free-tier 100 RPM ceiling (shared across every model/process
# hitting the same API key, so leave headroom rather than setting 100).
EMBEDDING_RPM_LIMIT = int(os.getenv("EMBEDDING_RPM_LIMIT", "80"))

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# --- Retrieval -------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))

# --- LLM ---------------------------------------------------------------
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# --- Testing -------------------------------------------------------------
RUN_LIVE_TESTS = os.getenv("RUN_LIVE_TESTS", "false").lower() in {"1", "true", "yes"}


def ensure_dirs() -> None:
    """Create data/vectorstore directories if they do not exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)