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


def remove(filename: str) -> None:
    with _LOCK:
        data = _load()
        data.pop(filename, None)
        _save(data)


def list_documents() -> List[dict]:
    """Every PDF currently on disk, with page count / upload time
    (when known) and a live `indexed` flag from the FAISS record
    manager.

    Self-heals drift between the metadata file and disk (a PDF
    dropped into data/ by hand, or removed outside the API): entries
    for files no longer on disk are dropped; files with no metadata
    entry get one created with `uploaded_at: None` (we don't know when
    -- it wasn't uploaded through this API).
    """
    # Local import: this module deliberately depends on vector_store
    # (to ask "is this file indexed"), not the other way around --
    # avoids a circular import at module load time.
    from src.vector_store import VectorStore

    with _LOCK:
        data = _load()
        on_disk = (
            sorted(p.name for p in config.DATA_DIR.glob("*.pdf"))
            if config.DATA_DIR.exists()
            else []
        )
        
        changed = False
        for filename in on_disk:
            if filename not in data:
                data[filename] = {
                    "filename": filename,
                    "pages": _page_count(config.DATA_DIR / filename),
                    "uploaded_at": None,
                }
                changed = True

        for stale in set(data) - set(on_disk):
            del data[stale]
            changed = True

        if changed:
            _save(data)

    vector_store = VectorStore()
    return [
        {
            "filename": filename,
            "pages": data[filename].get("pages"),
            "indexed": vector_store.is_indexed(filename),
            "uploaded_at": data[filename].get("uploaded_at"),
        }
        for filename in on_disk
    ]
