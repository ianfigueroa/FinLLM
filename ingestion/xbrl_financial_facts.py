from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser

_MODEL_LINE_ORDER = [
    "Revenue",
    "Gross profit",
    "Operating income",
    "Net income",
    "Cash and cash equivalents",
    "Total assets",
    "Total liabilities",
    "Total stockholders' equity",
    "Net cash provided by operating activities",
    "Capital expenditures",
    "Net cash from financing activities",
]
_ANNUAL_DURATION_DAYS = 250


@dataclass(frozen=True)
class _FinancialConcept:
    label: str
    period_type: str


_XBRL_FINANCIAL_CONCEPTS = {
    "us-gaap:revenues": _FinancialConcept("Revenue", "duration"),
    "us-gaap:revenuefromcontractwithcustomerexcludingassessedtax": _FinancialConcept(
        "Revenue", "duration"
    ),
    "us-gaap:salesrevenuenet": _FinancialConcept("Revenue", "duration"),
    "us-gaap:grossprofit": _FinancialConcept("Gross profit", "duration"),
    "us-gaap:operatingincomeloss": _FinancialConcept("Operating income", "duration"),
    "us-gaap:incomelossfromcontinuingoperationsbeforeincometaxes": _FinancialConcept(
        "Operating income", "duration"
    ),
    "us-gaap:netincomeloss": _FinancialConcept("Net income", "duration"),
    "us-gaap:profitloss": _FinancialConcept("Net income", "duration"),
    "us-gaap:cashandcashequivalentsatcarryingvalue": _FinancialConcept(
        "Cash and cash equivalents", "instant"
    ),
    "us-gaap:cashcashequivalentsrestrictedcashandrestrictedcashequivalents": (
        _FinancialConcept("Cash and cash equivalents", "instant")
    ),
    "us-gaap:assets": _FinancialConcept("Total assets", "instant"),
    "us-gaap:liabilities": _FinancialConcept("Total liabilities", "instant"),
    "us-gaap:stockholdersequity": _FinancialConcept("Total stockholders' equity", "instant"),
    "us-gaap:stockholdersequityincludingportionattributabletononcontrollinginterest": (
        _FinancialConcept("Total stockholders' equity", "instant")
    ),
    "us-gaap:netcashprovidedbyusedinoperatingactivities": _FinancialConcept(
        "Net cash provided by operating activities", "duration"
    ),
    "us-gaap:paymentstoacquirepropertyplantandequipment": _FinancialConcept(
        "Capital expenditures", "duration"
    ),
    "us-gaap:paymentstoacquirebusinessesnetofcashacquired": _FinancialConcept(
        "Capital expenditures", "duration"
    ),
    "us-gaap:paymentstoacquireproductiveassets": _FinancialConcept(
        "Capital expenditures", "duration"
    ),
    "us-gaap:netcashprovidedbyusedinfinancingactivities": _FinancialConcept(
        "Net cash from financing activities", "duration"
    ),
}


@dataclass
class _XbrlContext:
    context_id: str
    start_date: str = ""
    end_date: str = ""
    instant: str = ""
    has_dimensions: bool = False


@dataclass
class _XbrlFactDraft:
    concept: _FinancialConcept
    context_ref: str
    scale: int
    sign: str
    parts: list[str]


@dataclass(frozen=True)
class _XbrlFinancialFact:
    label: str
    context_ref: str
    period_type: str
    value: float


class _InlineXbrlFinancialExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.contexts: dict[str, _XbrlContext] = {}
        self.facts: list[_XbrlFinancialFact] = []
        self._active_context: _XbrlContext | None = None
        self._active_context_field: str | None = None
        self._active_context_parts: list[str] = []
        self._active_fact: _XbrlFactDraft | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        local_name = _local_name(tag)
        attributes = _attribute_dict(attrs)

        if local_name == "context":
            context_id = attributes.get("id")
            if context_id:
                self._active_context = _XbrlContext(context_id=context_id)
            return

        if self._active_context:
            if local_name in {"startdate", "enddate", "instant"}:
                self._active_context_field = local_name
                self._active_context_parts = []
                return
            if local_name in {"segment", "scenario", "explicitmember", "typedmember"}:
                self._active_context.has_dimensions = True

        if local_name != "nonfraction":
            return

        concept = _XBRL_FINANCIAL_CONCEPTS.get(attributes.get("name", "").lower())
        context_ref = attributes.get("contextref", "")
        if not concept or not context_ref:
            return

        self._active_fact = _XbrlFactDraft(
            concept=concept,
            context_ref=context_ref,
            scale=_parse_scale(attributes.get("scale")),
            sign=attributes.get("sign", ""),
            parts=[],
        )

    def handle_data(self, data: str) -> None:
        if self._active_context_field:
            self._active_context_parts.append(data)
        if self._active_fact:
            self._active_fact.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        local_name = _local_name(tag)
        if (
            self._active_context
            and self._active_context_field
            and local_name == self._active_context_field
        ):
            value = _compact_text(" ".join(self._active_context_parts))
            if self._active_context_field == "startdate":
                self._active_context.start_date = value
            elif self._active_context_field == "enddate":
                self._active_context.end_date = value
            elif self._active_context_field == "instant":
                self._active_context.instant = value
            self._active_context_field = None
            self._active_context_parts = []
            return

        if self._active_context and local_name == "context":
            self.contexts[self._active_context.context_id] = self._active_context
            self._active_context = None
            return

        if self._active_fact and local_name == "nonfraction":
            fact_value = _parse_fact_value(
                " ".join(self._active_fact.parts),
                scale=self._active_fact.scale,
                sign=self._active_fact.sign,
            )
            if fact_value is not None:
                self.facts.append(
                    _XbrlFinancialFact(
                        label=self._active_fact.concept.label,
                        context_ref=self._active_fact.context_ref,
                        period_type=self._active_fact.concept.period_type,
                        value=fact_value,
                    )
                )
            self._active_fact = None


def extract_inline_xbrl_financial_rows(html: str) -> str:
    extractor = _InlineXbrlFinancialExtractor()
    extractor.feed(html)
    values_by_label: dict[str, dict[str, float]] = defaultdict(dict)

    for fact in extractor.facts:
        context = extractor.contexts.get(fact.context_ref)
        if not context or context.has_dimensions:
            continue
        year = _financial_fact_year(context, fact.period_type)
        if not year:
            continue
        values_by_label[fact.label].setdefault(year, _round_fact_value(fact.value))

    rows = ["Inline XBRL financial facts"]
    for label in _MODEL_LINE_ORDER:
        values = values_by_label.get(label)
        if not values:
            continue
        row_values = " ".join(
            f"{year} {_format_fact_value(values[year])}"
            for year in sorted(values, reverse=True)
        )
        rows.append(f"{label} {row_values}")

    return "\n".join(rows) if len(rows) > 1 else ""


def _local_name(tag: str) -> str:
    return tag.rsplit(":", 1)[-1].lower()


def _attribute_dict(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
    return {name.lower(): value or "" for name, value in attrs}


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def _parse_scale(raw: str | None) -> int:
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _parse_fact_value(raw: str, *, scale: int, sign: str) -> float | None:
    cleaned = _compact_text(raw)
    if not cleaned or cleaned in {"-", "\u2014"}:
        return None
    negative = cleaned.startswith("(") and cleaned.endswith(")") or sign == "-"
    cleaned = cleaned.strip("()").replace("$", "").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if scale:
        value *= 10**scale
    return -value if negative else value


def _financial_fact_year(context: _XbrlContext, period_type: str) -> str | None:
    if period_type == "instant":
        return _year_from_date(context.instant or context.end_date)
    if not context.end_date:
        return None
    if context.start_date and _duration_days(context.start_date, context.end_date) < (
        _ANNUAL_DURATION_DAYS
    ):
        return None
    return _year_from_date(context.end_date)


def _year_from_date(value: str) -> str | None:
    match = re.match(r"^(20\d{2})-\d{2}-\d{2}$", value)
    return match.group(1) if match else None


def _duration_days(start_date: str, end_date: str) -> int:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return _ANNUAL_DURATION_DAYS
    return (end - start).days


def _round_fact_value(value: float) -> float:
    rounded = round(value, 2)
    return int(rounded) if rounded.is_integer() else rounded


def _format_fact_value(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}".rstrip("0").rstrip(".")
