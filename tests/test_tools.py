import pytest

from src.tools.calculator_tool import calculator, evaluate_expression
from src.tools.search_tool import make_search_documents_tool


def test_search_documents_returns_relevant_content(fake_retriever):
    tool = make_search_documents_tool(fake_retriever)
    result = tool.invoke({"query": "What is the company's revenue?"})
    assert "revenue" in result.lower() or "Document:" in result


def test_search_documents_includes_source_and_page(fake_retriever):
    tool = make_search_documents_tool(fake_retriever)
    result = tool.invoke({"query": "warranty"})
    assert "Document:" in result
    assert "Page:" in result


def test_calculator_multiplication():
    assert evaluate_expression("125 * 37") == 4625


def test_calculator_tool_invoke():
    result = calculator.invoke({"expression": "125 * 37"})
    assert result == "4625"


def test_calculator_rejects_unsafe_expressions():
    with pytest.raises(Exception):
        evaluate_expression("__import__('os').system('echo hi')")
