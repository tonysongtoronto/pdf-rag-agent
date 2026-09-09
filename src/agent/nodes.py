"""
Node implementations for the LangGraph agent:

- agent_node: asks the DeepSeek LLM (bound to tools) what to do next.
- tool_node: executes any tool calls the LLM requested, recording
  results (and, for search_documents, the retrieved documents) in state.
- should_continue: conditional-edge router between the two.
"""
from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.agent.state import AgentState
from src.llm import get_llm
from src.retriever import Retriever

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to two tools:\n"
    "- search_documents: search a PDF knowledge base. Use this whenever the "
    "user's question could be answered by information in the documents.\n"
    "- calculator: evaluate arithmetic expressions. Use this for any math "
    "instead of computing it yourself.\n"
    "If neither tool is needed (e.g. a greeting or general knowledge "
    "question), answer directly. When you used search_documents, mention "
    "which document/page the information came from."
)


def make_agent_node(tools: List[BaseTool]):
    # Bind the LLM to its tools lazily, on first actual invocation, so
    # constructing the graph never requires DEEPSEEK_API_KEY to already
    # be configured -- only running a question does.
    _bound_llm = {}

    def agent_node(state: AgentState) -> dict:
        if "llm" not in _bound_llm:
            _bound_llm["llm"] = get_llm().bind_tools(tools)

        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response: AIMessage = _bound_llm["llm"].invoke(messages)
        return {"messages": [response]}

    return agent_node


def make_tool_node(tools: List[BaseTool], retriever: Retriever | None = None):
    """Build the tool-executing node.

    `retriever` is optional and only used to recover structured
    {source, page} metadata for `search_documents` calls, so the final
    Agent.ask() response can report sources -- the tools themselves
    already ran and returned their (text) result to the LLM.
    """
    tools_by_name = {tool.name: tool for tool in tools}

    def tool_node(state: AgentState) -> dict:
        last_message = state["messages"][-1]
        tool_messages = []
        retrieved_documents = list(state.get("retrieved_documents", []))
        tool_results = list(state.get("tool_results", []))

        for call in getattr(last_message, "tool_calls", []) or []:
            tool = tools_by_name[call["name"]]
            result = tool.invoke(call["args"])
            tool_results.append({"tool": call["name"], "args": call["args"], "result": result})

            if call["name"] == "search_documents" and retriever is not None:
                query = call["args"].get("query", "")
                retrieved_documents.extend(retriever.invoke(query))

            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"])
            )

        return {
            "messages": tool_messages,
            "retrieved_documents": retrieved_documents,
            "tool_results": tool_results,
        }

    return tool_node


def should_continue(state: AgentState) -> str:
    """Route to the tool node if the last AI message requested tool calls."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return "end"
