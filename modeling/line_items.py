from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LineDefinition:
    statement: str
    key: str
    label: str
    patterns: tuple[str, ...]


LINE_DEFINITIONS = [
    LineDefinition(
        "income_statement",
        "revenue",
        "Revenue",
        (r"^(total\s+)?(net\s+)?revenues?$", r"^net sales$"),
    ),
    LineDefinition("income_statement", "gross_profit", "Gross profit", (r"^gross profit$",)),
    LineDefinition(
        "income_statement",
        "operating_income",
        "Operating income",
        (r"^operating income$", r"^income from operations$"),
    ),
    LineDefinition(
        "income_statement",
        "net_income",
        "Net income",
        (r"^net income$", r"^net earnings$"),
    ),
    LineDefinition(
        "balance_sheet",
        "cash_and_equivalents",
        "Cash and cash equivalents",
        (r"^cash and cash equivalents$", r"^cash, cash equivalents"),
    ),
    LineDefinition("balance_sheet", "total_assets", "Total assets", (r"^total assets$",)),
    LineDefinition(
        "balance_sheet",
        "total_liabilities",
        "Total liabilities",
        (r"^total liabilities$",),
    ),
    LineDefinition(
        "balance_sheet",
        "stockholders_equity",
        "Stockholders' equity",
        (r"^(total\s+)?stockholders[’']?\s+equity$", r"^shareholders[’']?\s+equity$"),
    ),
    LineDefinition(
        "cash_flow_statement",
        "operating_cash_flow",
        "Net cash provided by operating activities",
        (r"^net cash (provided by|from) operating activities$",),
    ),
    LineDefinition(
        "cash_flow_statement",
        "capital_expenditures",
        "Capital expenditures",
        (r"^capital expenditures$", r"^purchases of property and equipment$"),
    ),
    LineDefinition(
        "cash_flow_statement",
        "financing_cash_flow",
        "Net cash from financing activities",
        (r"^net cash (used in|provided by|from) financing activities$",),
    ),
]

STATEMENT_LABELS = {
    "income_statement": "Income statement",
    "balance_sheet": "Balance sheet",
    "cash_flow_statement": "Cash flow statement",
}
