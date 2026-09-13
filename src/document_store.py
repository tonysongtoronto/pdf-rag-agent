"""
Lightweight metadata for GET /documents display: filename, page
count, upload time. Nothing here tracks "is this file indexed" any
more -- that used to be a hand-rolled boolean, but it's now answered
live by asking LangChain's RecordManager (via
VectorStore.is_indexed(), see src/vector_store.py) which source ids
it actually has records for. One source of truth instead of two that
can drift apart.

Not a database -- fine for a single local backend with one writer.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader

from src import config


_LOCK = threading.Lock()


def _metadata_path() -> Path:
    return config.VECTORSTORE_DIR / "documents_metadata.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _page_count(path: Path) -> Optional[int]:
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        # Corrupt/unreadable PDF -- don't let metadata bookkeeping
        # crash the request over it, just report an unknown page count.
        return None


def _load() -> Dict[str, dict]:
    path = _metadata_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: Dict[str, dict]) -> None:
    config.ensure_dirs()
    _metadata_path().write_text(json.dumps(data, indent=2, sort_keys=True))


def register_upload(filename: str) -> None:
    """Record a freshly-uploaded file's page count and upload time."""
    with _LOCK:
        data = _load()
        data[filename] = {
            "filename": filename,
            "pages": _page_count(config.DATA_DIR / filename),
            "uploaded_at": _now(),
        }
        _save(data)


def get(filename: str) -> Optional[dict]:
    """The metadata record for `filename`, or None if it isn't tracked.
    Used by the upload route to tell "new file" from "retrying an
    upload that was never indexed" from "already exists and is
    indexed" -- see src/api/routes.py."""
    with _LOCK:
        data = _load()
    return data.get(filename)


def remove(filename: str) -> None:
    with _LOCK:
        data = _load()
        data.pop(filename, None)
        _save(data)


def list_documents() -> List[dict]:
    """Every document this metadata file knows about, with page count
    / upload time (when known) and a live `indexed` flag from the
    FAISS record manager.

    This metadata file -- not DATA_DIR -- is the source of truth for
    "what documents exist". DATA_DIR is a temporary staging area used
    only while a file is being uploaded/indexed; a deploy with
    non-persistent local disk can lose everything in it between
    restarts without that meaning any document was "deleted". A file
    disappearing from disk is only ever noticed (and only matters) at
    the point something tries to read it, e.g. POST
    /documents/{filename}/index -- not here.
    """
    # Local import: this module deliberately depends on vector_store
    # (to ask "is this file indexed"), not the other way around --
    # avoids a circular import at module load time.
    from src.vector_store import VectorStore

    with _LOCK:
        data = _load()

    vector_store = VectorStore()
    return [
        {
            "filename": filename,
            "pages": meta.get("pages"),
            "indexed": vector_store.is_indexed(filename),
            "uploaded_at": meta.get("uploaded_at"),
        }
        for filename, meta in sorted(data.items())
    ]