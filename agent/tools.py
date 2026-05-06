from __future__ import annotations

import ast
import operator
import sqlite3
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


class SQLMetadataTool(BaseTool):
    name = "metadata_sql"
    description = "Run read-only SQL over indexed document metadata."
    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string", "maxLength": 1_000}},
    }

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def run(self, payload: dict[str, Any]) -> ToolResult:
        query = str(payload.get("query", "")).strip()
        if not _is_read_only_select(query):
            return ToolResult(
                ok=False, error="metadata_sql is read-only and only accepts SELECT queries"
            )

        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        try:
            create_sql = (
                "CREATE TABLE documents ("
                "ticker TEXT, company TEXT, form_type TEXT, filing_date TEXT, "
                "section TEXT, source TEXT)"
            )
            insert_sql = (
                "INSERT INTO documents VALUES "
                "(:ticker, :company, :form_type, :filing_date, :section, :source)"
            )
            connection.execute(create_sql)
            connection.executemany(
                insert_sql,
                [_metadata_row(row) for row in self._rows],
            )
            rows = connection.execute(query).fetchall()
            return ToolResult(ok=True, output=[dict(row) for row in rows])
        except sqlite3.Error:
            return ToolResult(ok=False, error="metadata query failed")
        finally:
            connection.close()


class MarketDataTool(BaseTool):
    name = "market_data"
    description = "Return cached local market prices for a ticker."
    input_schema = {
        "type": "object",
        "required": ["ticker"],
        "properties": {"ticker": {"type": "string", "maxLength": 12}},
    }

    def __init__(self, prices_by_ticker: dict[str, list[dict[str, float | str]]]) -> None:
        self._prices_by_ticker = {
            ticker.upper(): prices for ticker, prices in prices_by_ticker.items()
        }

    def run(self, payload: dict[str, Any]) -> ToolResult:
        ticker = str(payload.get("ticker", "")).upper()
        prices = self._prices_by_ticker.get(ticker)
        if prices is None:
            return ToolResult(ok=False, error=f"no cached market data for {ticker}")
        return ToolResult(ok=True, output={"ticker": ticker, "prices": prices})


class BacktestTool(BaseTool):
    name = "simple_backtest"
    description = "Backtest a long-only threshold strategy using local prices and dated scores."
    input_schema = {
        "type": "object",
        "required": ["prices", "signals", "threshold"],
        "properties": {
            "prices": {"type": "array"},
            "signals": {"type": "array"},
            "threshold": {"type": "number"},
        },
    }

    def run(self, payload: dict[str, Any]) -> ToolResult:
        try:
            prices = payload["prices"]
            signals = {signal["date"]: float(signal["score"]) for signal in payload["signals"]}
            threshold = float(payload["threshold"])
            cumulative_return = 0.0
            trades = 0
            for current, following in zip(prices, prices[1:], strict=False):
                if signals.get(current["date"], 0.0) < threshold:
                    continue
                cumulative_return += (following["close"] - current["close"]) / current["close"]
                trades += 1
            return ToolResult(
                ok=True,
                output={"cumulative_return": cumulative_return, "trades": trades},
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            return ToolResult(ok=False, error=str(exc))


class PythonAnalysisTool(BaseTool):
    name = "python_analysis"
    description = "Run a narrow Python subset for simple list and arithmetic analysis."
    input_schema = {
        "type": "object",
        "required": ["code"],
        "properties": {"code": {"type": "string", "maxLength": 1_000}},
    }

    _allowed_nodes = (
        ast.Module,
        ast.Assign,
        ast.Expr,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.BinOp,
        ast.UnaryOp,
        ast.Subscript,
        ast.Slice,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
    )
    _allowed_functions = {"sum": sum, "min": min, "max": max, "len": len, "round": round}

    def run(self, payload: dict[str, Any]) -> ToolResult:
        code = str(payload.get("code", ""))
        try:
            tree = ast.parse(code, mode="exec")
            _validate_python_subset(tree, self._allowed_nodes, set(self._allowed_functions))
            namespace: dict[str, Any] = {}
            exec(
                compile(tree, "<finllm-python-analysis>", "exec"),
                {"__builtins__": self._allowed_functions},
                namespace,
            )
            return ToolResult(ok=True, output=namespace.get("result"))
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as exc:
            return ToolResult(ok=False, error=str(exc))


def _eval_arithmetic(node: ast.AST, operators: dict[type[ast.AST], Any]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp):
        operator_fn = operators.get(type(node.op))
        if operator_fn is None:
            raise ValueError("unsupported arithmetic operator")
        return float(
            operator_fn(
                _eval_arithmetic(node.left, operators),
                _eval_arithmetic(node.right, operators),
            )
        )
    if isinstance(node, ast.UnaryOp):
        operator_fn = operators.get(type(node.op))
        if operator_fn is None:
            raise ValueError("unsupported unary operator")
        return float(operator_fn(_eval_arithmetic(node.operand, operators)))
    raise ValueError("unsupported expression")


def _positive_denominator(payload: dict[str, Any], field: str) -> float:
    value = payload[field]
    if value == 0:
        raise ValueError(f"{field} must be non-zero")
    return float(value)


def _is_read_only_select(query: str) -> bool:
    normalized = query.rstrip(";").strip().lower()
    blocked = ["insert", "update", "delete", "drop", "alter", "pragma", "attach", "detach"]
    return normalized.startswith("select ") and not any(
        word in normalized.split() for word in blocked
    )


def _metadata_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": row.get("ticker"),
        "company": row.get("company"),
        "form_type": row.get("form_type"),
        "filing_date": row.get("filing_date"),
        "section": row.get("section"),
        "source": row.get("source"),
    }


def _validate_python_subset(
    tree: ast.AST,
    allowed_nodes: tuple[type[ast.AST], ...],
    allowed_functions: set[str],
) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"unsupported python syntax: {type(node).__name__}")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in allowed_functions
        ):
            raise ValueError("unsupported python function call")
