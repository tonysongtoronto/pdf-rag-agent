"""
Simple arithmetic tool so the Agent doesn't have to (mis)calculate
numbers itself.
"""
from __future__ import annotations

import ast
import operator

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def evaluate_expression(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression (+, -, *, /, %, **)."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)


class CalculatorInput(BaseModel):
    expression: str = Field(description="A basic arithmetic expression, e.g. '125 * 37'.")


def _run(expression: str) -> str:
    result = evaluate_expression(expression)
    # Return integers without a trailing ".0" for cleaner display.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


calculator = StructuredTool.from_function(
    func=_run,
    name="calculator",
    description="Evaluate a basic arithmetic expression such as '125 * 37'.",
    args_schema=CalculatorInput,
)
