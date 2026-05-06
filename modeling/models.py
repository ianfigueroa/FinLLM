from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSource:
    marker: str
    chunk_id: str
    source: str
    source_url: str | None
    section: str | None
    excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "source_url": self.source_url,
            "section": self.section,
            "excerpt": self.excerpt,
        }


@dataclass
class StatementLine:
    label: str
    historical: dict[str, float] = field(default_factory=dict)
    sources: list[ModelSource] = field(default_factory=list)
    formula: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "historical": self.historical,
            "sources": [source.to_dict() for source in self.sources],
            "formula": self.formula,
        }


@dataclass
class StatementTable:
    label: str
    lines: dict[str, StatementLine] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "lines": {key: line.to_dict() for key, line in self.lines.items()},
        }


@dataclass
class ThreeStatementModel:
    ticker: str
    company: str | None
    form_type: str | None
    filing_date: str | None
    statements: dict[str, StatementTable]
    projections: dict[str, dict[str, dict[str, float]]]
    assumptions: dict[str, float]
    sources: list[ModelSource]
    limitations: list[str]
    confidence: float
    disclaimer: str = "Research analysis only; not financial advice."

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "company": self.company,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "statements": {
                key: statement.to_dict() for key, statement in self.statements.items()
            },
            "projections": self.projections,
            "assumptions": self.assumptions,
            "sources": [source.to_dict() for source in self.sources],
            "limitations": self.limitations,
            "confidence": self.confidence,
            "disclaimer": self.disclaimer,
        }
