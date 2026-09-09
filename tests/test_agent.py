import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.agent.graph import Agent
from src.agent.state import AgentState
from src.agent.nodes import should_continue

requires_live = pytest.mark.skipif(
    not (config.RUN_LIVE_TESTS and config.DEEPSEEK_API_KEY),
    reason="Set RUN_LIVE_TESTS=true and DEEPSEEK_API_KEY to run live agent tests.",
)


def _state_with(message) -> AgentState:
    return {
        "messages": [HumanMessage(content="hi"), message],
        "question": "hi",
        "retrieved_documents": [],
        "tool_results": [],
        "final_answer": "",
    }


def test_should_continue_routes_to_tools_when_tool_calls_present():
    message = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "call_1"}],
    )
    assert should_continue(_state_with(message)) == "tools"


def test_should_continue_routes_to_end_without_tool_calls():
    message = AIMessage(content="Hello! How can I help?")
    assert should_continue(_state_with(message)) == "end"


@requires_live
def test_agent_no_tool_needed_for_greeting(fake_retriever):
    agent = Agent(retriever=fake_retriever)
    response = agent.ask("Hello")
    assert response["answer"]
    assert response["sources"] == []


@requires_live
def test_agent_uses_search_documents_for_knowledge_question(fake_retriever):
    agent = Agent(retriever=fake_retriever)
    response = agent.ask("What does the document say about revenue?")
    assert response["answer"]
    assert len(response["sources"]) > 0


@requires_live
def test_agent_uses_calculator_for_math_question(fake_retriever):
    agent = Agent(retriever=fake_retriever)
    response = agent.ask("What is 125 * 37?")
    assert "4625" in response["answer"]
