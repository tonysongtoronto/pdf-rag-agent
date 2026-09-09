"""
LangGraph state schema shared by every node in the agent graph.
"""
from __future__ import annotations

from typing import Annotated, Any, List, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    retrieved_documents: List[Any]
    tool_results: List[Any]
    final_answer: str
