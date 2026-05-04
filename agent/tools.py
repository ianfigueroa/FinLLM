from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str = ""


class BaseTool:
    name: str
    description: str
    input_schema: dict[str, Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluate simple arithmetic expressions for financial calculations."
    input_schema = {
        "type": "object",
        "required": ["expression"],
        "properties": {"expression": {"type": "string", "maxLength": 200}},
    }

    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def run(self, payload: dict[str, Any]) -> ToolResult:
        expression = str(payload.get("expression", ""))
        try:
            tree = ast.parse(expression, mode="eval")
            return ToolResult(ok=True, output=_eval_arithmetic(tree.body, self._operators))
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
            return ToolResult(ok=False, error=str(exc))


class FinancialRatioTool(BaseTool):
    name = "financial_ratio_calculator"
    description = "Calculate current ratio, debt-to-equity, and net margin from statement inputs."
    input_schema = {
        "type": "object",
        "required": [
            "current_assets",
            "current_liabilities",
            "total_debt",
            "shareholders_equity",
            "net_income",
            "revenue",
        ],
        "properties": {
            "current_assets": {"type": "number"},
            "current_liabilities": {"type": "number"},
            "total_debt": {"type": "number"},
            "shareholders_equity": {"type": "number"},
            "net_income": {"type": "number"},
            "revenue": {"type": "number"},
        },
    }

    def run(self, payload: dict[str, Any]) -> ToolResult:
        try:
            current_liabilities = _positive_denominator(payload, "current_liabilities")
            equity = _positive_denominator(payload, "shareholders_equity")
            revenue = _positive_denominator(payload, "revenue")
            output = {
                "current_ratio": payload["current_assets"] / current_liabilities,
                "debt_to_equity": payload["total_debt"] / equity,
                "net_margin": payload["net_income"] / revenue,
            }
            return ToolResult(ok=True, output=output)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(ok=False, error=str(exc))


def _eval_arithmetic(node: ast.AST, operators: dict[type[ast.AST], Any]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp):
        operator_fn = operators.get(type(node.op))
        if operator_fn is None:
            raise ValueError("unsupported arithmetic operator")
        return operator_fn(_eval_arithmetic(node.left, operators), _eval_arithmetic(node.right, operators))
    if isinstance(node, ast.UnaryOp):
        operator_fn = operators.get(type(node.op))
        if operator_fn is None:
            raise ValueError("unsupported unary operator")
        return operator_fn(_eval_arithmetic(node.operand, operators))
    raise ValueError("unsupported expression")


def _positive_denominator(payload: dict[str, Any], field: str) -> float:
    value = payload[field]
    if value == 0:
        raise ValueError(f"{field} must be non-zero")
    return float(value)
