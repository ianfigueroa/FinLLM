from __future__ import annotations

import re
from collections.abc import Iterable

from ingestion.metadata import DocumentChunk
from modeling.line_items import LINE_DEFINITIONS, STATEMENT_LABELS, LineDefinition
from modeling.models import ModelSource, StatementLine, StatementTable, ThreeStatementModel

_YEAR_VALUE = re.compile(
    r"(?P<year>20\d{2})\s+\$?(?P<number>\(?-?[\d,]+(?:\.\d+)?\)?)",
    re.IGNORECASE,
)

_DEFAULT_REVENUE_GROWTH = 0.05


def build_three_statement_model(
    chunks: Iterable[DocumentChunk],
    *,
    ticker: str,
    projection_years: int = 3,
    revenue_growth: float = _DEFAULT_REVENUE_GROWTH,
) -> ThreeStatementModel:
    bounded_projection_years = max(1, min(projection_years, 5))
    filtered_chunks = [
        chunk for chunk in chunks if (chunk.metadata.ticker or "").upper() == ticker.upper()
    ]
    statements = _empty_statements()
    sources: list[ModelSource] = []

    for chunk in filtered_chunks:
        for row in _candidate_rows(chunk.text):
            definition = _match_definition(row)
            if not definition:
                continue
            values = _extract_year_values(row)
            if not values:
                continue
            source = _source_from_chunk(chunk, row, len(sources) + 1)
            sources.append(source)
            line = statements[definition.statement].lines.setdefault(
                definition.key,
                StatementLine(label=definition.label),
            )
            _merge_values(line.historical, values)
            if source not in line.sources:
                line.sources.append(source)

    _derive_free_cash_flow(statements, sources)
    projections = _build_projections(statements, bounded_projection_years, revenue_growth)
    metadata_source = filtered_chunks[0].metadata if filtered_chunks else None
    limitations = _limitations(statements, projections, filtered_chunks)
    return ThreeStatementModel(
        ticker=ticker.upper(),
        company=metadata_source.company if metadata_source else None,
        form_type=metadata_source.form_type if metadata_source else None,
        filing_date=metadata_source.filing_date if metadata_source else None,
        statements=statements,
        projections=projections,
        assumptions={
            "revenue_growth": revenue_growth,
            "projection_years": float(bounded_projection_years),
        },
        sources=sources,
        limitations=limitations,
        confidence=_confidence(statements, sources, projections),
    )


def _empty_statements() -> dict[str, StatementTable]:
    return {
        key: StatementTable(label=label)
        for key, label in STATEMENT_LABELS.items()
    }


def _candidate_rows(text: str) -> list[str]:
    rows: list[str] = []
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if len(normalized) >= 8:
            rows.append(normalized)
    return rows


def _match_definition(row: str) -> LineDefinition | None:
    label = _row_label(row)
    for definition in LINE_DEFINITIONS:
        if any(re.search(pattern, label, re.IGNORECASE) for pattern in definition.patterns):
            return definition
    return None


def _row_label(row: str) -> str:
    year_match = re.search(r"\b20\d{2}\b", row)
    label = row[: year_match.start()] if year_match else row
    label = re.sub(r"\s+", " ", label).strip(" :-")
    return label.lower()


def _extract_year_values(row: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for match in _YEAR_VALUE.finditer(row):
        number = _parse_number(match.group("number"))
        if number is not None:
            values[match.group("year")] = number
    return values


def _parse_number(raw: str) -> float | None:
    cleaned = raw.replace("$", "").replace(",", "").strip()
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def _source_from_chunk(chunk: DocumentChunk, row: str, index: int) -> ModelSource:
    metadata = chunk.metadata
    return ModelSource(
        marker=f"M{index}",
        chunk_id=chunk.chunk_id,
        source=metadata.source,
        source_url=metadata.source_url,
        section=metadata.section,
        excerpt=row[:280],
    )


def _merge_values(target: dict[str, float], values: dict[str, float]) -> None:
    for year, value in values.items():
        target.setdefault(year, _round_number(value))


def _derive_free_cash_flow(
    statements: dict[str, StatementTable],
    sources: list[ModelSource],
) -> None:
    cash_flow = statements["cash_flow_statement"].lines
    operating = cash_flow.get("operating_cash_flow")
    capex = cash_flow.get("capital_expenditures")
    if not operating or not capex:
        return

    years = set(operating.historical) & set(capex.historical)
    if not years:
        return
    values = {}
    for year in years:
        values[year] = _free_cash_flow_value(
            operating.historical[year],
            capex.historical[year],
        )
    source_pool = [*operating.sources, *capex.sources]
    cash_flow["free_cash_flow"] = StatementLine(
        label="Free cash flow",
        historical={year: _round_number(value) for year, value in values.items()},
        sources=source_pool,
        formula="operating_cash_flow - capital_expenditures",
    )
    for source in source_pool:
        if source not in sources:
            sources.append(source)


def _build_projections(
    statements: dict[str, StatementTable],
    projection_years: int,
    revenue_growth: float,
) -> dict[str, dict[str, dict[str, float]]]:
    latest_year = _latest_year(statements)
    revenue_line = statements["income_statement"].lines.get("revenue")
    if latest_year is None or not revenue_line or latest_year not in revenue_line.historical:
        return {}

    projections: dict[str, dict[str, dict[str, float]]] = {}
    revenue = revenue_line.historical[latest_year]
    prior_balance = _latest_values(statements["balance_sheet"])
    margins = _latest_margins(statements, latest_year, revenue)

    for offset in range(1, projection_years + 1):
        year = str(int(latest_year) + offset)
        revenue = _round_number(revenue * (1 + revenue_growth))
        income_statement = {"revenue": revenue}
        for key in ("gross_profit", "operating_income", "net_income"):
            if key in margins:
                income_statement[key] = _round_number(revenue * margins[key])

        cash_flow_statement: dict[str, float] = {}
        if "operating_cash_flow" in margins:
            cash_flow_statement["operating_cash_flow"] = _round_number(
                revenue * margins["operating_cash_flow"]
            )
        if "capital_expenditures" in margins:
            cash_flow_statement["capital_expenditures"] = _round_number(
                revenue * margins["capital_expenditures"]
            )
        if {"operating_cash_flow", "capital_expenditures"} <= set(cash_flow_statement):
            cash_flow_statement["free_cash_flow"] = _round_number(
                _free_cash_flow_value(
                    cash_flow_statement["operating_cash_flow"],
                    cash_flow_statement["capital_expenditures"],
                )
            )

        balance_sheet = _project_balance_sheet(prior_balance, revenue_growth)
        prior_balance = balance_sheet or prior_balance
        projections[year] = {
            "income_statement": income_statement,
            "balance_sheet": balance_sheet,
            "cash_flow_statement": cash_flow_statement,
        }
    return projections


def _latest_year(statements: dict[str, StatementTable]) -> str | None:
    years = {
        int(year)
        for statement in statements.values()
        for line in statement.lines.values()
        for year in line.historical
        if year.isdigit()
    }
    return str(max(years)) if years else None


def _latest_values(statement: StatementTable) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, line in statement.lines.items():
        if not line.historical:
            continue
        year = max(line.historical, key=int)
        values[key] = line.historical[year]
    return values


def _latest_margins(
    statements: dict[str, StatementTable],
    latest_year: str,
    revenue: float,
) -> dict[str, float]:
    if revenue == 0:
        return {}
    margins: dict[str, float] = {}
    for statement_key in ("income_statement", "cash_flow_statement"):
        for key, line in statements[statement_key].lines.items():
            if key == "revenue" or latest_year not in line.historical:
                continue
            margins[key] = line.historical[latest_year] / revenue
    return margins


def _project_balance_sheet(
    latest: dict[str, float],
    revenue_growth: float,
) -> dict[str, float]:
    if not latest:
        return {}
    projected = {
        key: _round_number(value * (1 + revenue_growth))
        for key, value in latest.items()
        if key != "stockholders_equity"
    }
    if {"total_assets", "total_liabilities"} <= set(projected):
        projected["stockholders_equity"] = _round_number(
            projected["total_assets"] - projected["total_liabilities"]
        )
    elif "stockholders_equity" in latest:
        projected["stockholders_equity"] = _round_number(
            latest["stockholders_equity"] * (1 + revenue_growth)
        )
    return projected


def _limitations(
    statements: dict[str, StatementTable],
    projections: dict[str, dict[str, dict[str, float]]],
    chunks: list[DocumentChunk],
) -> list[str]:
    limitations: list[str] = []
    if not chunks:
        limitations.append("Index documents for the selected ticker before building a model.")
        return limitations
    missing = [
        statement.label
        for statement in statements.values()
        if not statement.lines
    ]
    if missing:
        limitations.append("Missing extracted financial statement evidence: " + ", ".join(missing))
    if not projections:
        limitations.append("Projection scaffold requires at least a cited revenue line.")
    limitations.append("Outputs are a source-backed modeling scaffold, not audited financials.")
    return limitations


def _confidence(
    statements: dict[str, StatementTable],
    sources: list[ModelSource],
    projections: dict[str, dict[str, dict[str, float]]],
) -> float:
    line_count = sum(len(statement.lines) for statement in statements.values())
    statement_coverage = sum(1 for statement in statements.values() if statement.lines)
    score = min(line_count / 10, 1.0) * 0.45
    score += (statement_coverage / 3) * 0.35
    score += min(len(sources) / max(line_count, 1), 1.0) * 0.1
    score += 0.1 if projections else 0.0
    return round(min(score, 0.95), 4)


def _round_number(value: float) -> float:
    rounded = round(value, 2)
    return int(rounded) if rounded.is_integer() else rounded


def _free_cash_flow_value(operating_cash_flow: float, capital_expenditures: float) -> float:
    if capital_expenditures > 0:
        return operating_cash_flow - capital_expenditures
    return operating_cash_flow + capital_expenditures
