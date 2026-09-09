"""
Wraps the Retriever as an Agent-callable Tool: `search_documents`.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.rag import format_documents_for_context
from src.retriever import Retriever


class SearchDocumentsInput(BaseModel):
    query: str = Field(description="The question or topic to search for in the PDF knowledge base.")


def make_search_documents_tool(retriever: Optional[Retriever] = None) -> StructuredTool:
    """Build a `search_documents` tool bound to a specific retriever.

    Kept as a factory (rather than a single module-level tool) so tests
    can inject a fake retriever instead of touching FAISS/Gemini.
    """
    _retriever = retriever or Retriever()

    def _run(query: str) -> str:
        documents = _retriever.invoke(query)
        return format_documents_for_context(documents)

    return StructuredTool.from_function(
        func=_run,
        name="search_documents",
        description=(
            "Search the PDF knowledge base for information relevant to the "
            "user's question. Returns matching passages along with their "
            "source filename and page number."
        ),
        args_schema=SearchDocumentsInput,
    )


# Default, lazily-bound tool instance for convenience imports. The
# underlying Retriever/VectorStore is only touched the first time the
# tool actually runs, so importing this module never requires FAISS/Gemini
# to already be available.
search_documents = make_search_documents_tool()
