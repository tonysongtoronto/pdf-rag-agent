"""
Builds the LangGraph state graph:

    START -> agent -> (tools? -> agent)* -> END

and exposes a small `Agent` class with an `.ask(question)` interface
that returns `{"answer": ..., "sources": [...]}`.
"""
from __future__ import annotations

from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import make_agent_node, make_tool_node, should_continue
from src.agent.state import AgentState
from src.rag import documents_to_sources
from src.retriever import Retriever
from src.tools.calculator_tool import calculator
from src.tools.search_tool import make_search_documents_tool


def build_agent_graph(tools: List[BaseTool], retriever: Optional[Retriever] = None):
    """Compile the LangGraph agent graph for the given tool set."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(tools))
    graph.add_node("tools", make_tool_node(tools, retriever=retriever))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


class Agent:
    """High-level, ready-to-use PDF RAG + Tool agent."""

    def __init__(self, retriever: Optional[Retriever] = None, tools: Optional[List[BaseTool]] = None) -> None:
        self.retriever = retriever or Retriever()
        self.tools = tools or [make_search_documents_tool(self.retriever), calculator]
        self.graph = build_agent_graph(self.tools, retriever=self.retriever)

    def ask(self, question: str) -> dict:
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "question": question,
                "retrieved_documents": [],
                "tool_results": [],
                "final_answer": "",
            }
        )

        final_message = result["messages"][-1]
        answer = final_message.content if isinstance(final_message, AIMessage) else str(final_message.content)
        sources = documents_to_sources(result.get("retrieved_documents", []))

        return {"answer": answer, "sources": sources}
