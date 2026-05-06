from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ingestion.document_cleaner import clean_text
from ingestion.metadata import Document, DocumentMetadata

_SEC_HOSTS = {"sec.gov", "www.sec.gov"}
_SEC_ARCHIVE_PREFIX = "/Archives/edgar/data/"
_SEC_DOCUMENT_SUFFIXES = (".htm", ".html", ".txt")
_MAX_FILING_BYTES = 10 * 1024 * 1024
_DATE_IN_FILENAME = re.compile(r"-(\d{8})(?:\.[^.]+)?$")
_TARGET_DEI_FIELDS = {
    "dei:documenttype": "form_type",
    "dei:documentperiodenddate": "filing_date",
    "dei:entityregistrantname": "company",
    "dei:tradingsymbol": "ticker",
}


@dataclass(frozen=True)
class SecFilingMetadata:
    source_url: str
    ticker: str = ""
    company: str = ""
    form_type: str = ""
    filing_date: str = ""


class _SecHtmlTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag in {"br", "div", "p", "tr", "li", "table", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"div", "p", "tr", "li", "table", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)

    def text(self) -> str:
        return clean_text(" ".join(self._parts))


class _SecInlineMetadataExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: dict[str, list[str]] = {
            "ticker": [],
            "company": [],
            "form_type": [],
            "filing_date": [],
        }
        self._active_field: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "ix:nonnumeric":
            return
        name = (dict(attrs).get("name") or "").lower()
        field = _TARGET_DEI_FIELDS.get(name)
        if field:
            self._active_field = field
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._active_field:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "ix:nonnumeric" or not self._active_field:
            return
        value = clean_text(" ".join(self._parts))
        if value:
            self.values[self._active_field].append(value)
        self._active_field = None
        self._parts = []


def normalize_sec_filing_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() not in _SEC_HOSTS:
        raise ValueError("Only SEC archive filing URLs are supported.")

    if parsed.path == "/ix":
        doc = parse_qs(parsed.query).get("doc", [""])[0]
        archive_path = unquote(doc)
    else:
        archive_path = unquote(parsed.path)

    if not archive_path.startswith(_SEC_ARCHIVE_PREFIX) or not archive_path.endswith(
        _SEC_DOCUMENT_SUFFIXES
    ):
        raise ValueError("Only SEC archive filing documents are supported.")

    return f"https://www.sec.gov{archive_path}"


def extract_sec_html_text(html: str) -> str:
    extractor = _SecHtmlTextExtractor()
    extractor.feed(html)
    return extractor.text()


def extract_sec_filing_metadata(html: str, *, source_url: str) -> SecFilingMetadata:
    normalized_url = normalize_sec_filing_url(source_url)
    extractor = _SecInlineMetadataExtractor()
    extractor.feed(html)
    values = extractor.values
    filename_ticker = _ticker_from_filename(normalized_url)
    ticker = _preferred_ticker(filename_ticker, values["ticker"])
    filing_date = _date_to_iso(_first(values["filing_date"])) or _date_from_filename(
        normalized_url
    )

    return SecFilingMetadata(
        source_url=normalized_url,
        ticker=ticker,
        company=_first(values["company"]),
        form_type=_first(values["form_type"]).upper(),
        filing_date=filing_date,
    )


def fetch_sec_filing_metadata(url: str) -> SecFilingMetadata:
    normalized_url, html = _fetch_sec_filing_html(url)
    return extract_sec_filing_metadata(html, source_url=normalized_url)


def load_sec_filing_from_html(
    html: str,
    *,
    source_url: str,
    ticker: str = "",
    company: str = "",
    form_type: str = "",
    filing_date: str = "",
) -> Document:
    normalized_url = normalize_sec_filing_url(source_url)
    detected = extract_sec_filing_metadata(html, source_url=normalized_url)
    return Document(
        text=extract_sec_html_text(html),
        metadata=DocumentMetadata(
            source=normalized_url,
            ticker=(detected.ticker or ticker).upper(),
            company=detected.company or company,
            form_type=(detected.form_type or form_type).upper(),
            filing_date=detected.filing_date or filing_date,
            source_url=normalized_url,
        ),
    )


def load_sec_filing_url(
    url: str,
    *,
    ticker: str = "",
    company: str = "",
    form_type: str = "",
    filing_date: str = "",
) -> Document:
    normalized_url, html = _fetch_sec_filing_html(url)
    return load_sec_filing_from_html(
        html,
        source_url=normalized_url,
        ticker=ticker,
        company=company,
        form_type=form_type,
        filing_date=filing_date,
    )


def _fetch_sec_filing_html(url: str) -> tuple[str, str]:
    normalized_url = normalize_sec_filing_url(url)
    user_agent = os.getenv(
        "FINLLM_SEC_USER_AGENT",
        "FinLLMResearchAgent/0.1 local-research set-FINLLM_SEC_USER_AGENT",
    )
    with httpx.Client(
        timeout=20,
        follow_redirects=True,
        headers={"User-Agent": user_agent, "Accept": "text/html,text/plain"},
    ) as client:
        response = client.get(normalized_url)
        response.raise_for_status()

    if len(response.content) > _MAX_FILING_BYTES:
        raise ValueError("SEC filing exceeds 10 MB limit.")
    normalize_sec_filing_url(str(response.url))
    return normalized_url, response.text


def _first(values: list[str]) -> str:
    return values[0] if values else ""


def _ticker_from_filename(source_url: str) -> str:
    filename = PurePosixPath(urlparse(source_url).path).name
    stem = filename.rsplit(".", 1)[0]
    prefix = stem.split("-", 1)[0]
    return prefix.upper() if prefix else ""


def _preferred_ticker(filename_ticker: str, detected_tickers: list[str]) -> str:
    normalized_tickers = [ticker.upper() for ticker in detected_tickers if ticker]
    if filename_ticker and filename_ticker in normalized_tickers:
        return filename_ticker
    if normalized_tickers:
        return normalized_tickers[0]
    return filename_ticker


def _date_from_filename(source_url: str) -> str:
    filename = PurePosixPath(urlparse(source_url).path).name
    match = _DATE_IN_FILENAME.search(filename)
    if not match:
        return ""
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _date_to_iso(value: str) -> str:
    normalized = value.replace("\xa0", " ").strip()
    if not normalized:
        return ""
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(normalized, fmt).date().isoformat()
        except ValueError:
            continue
    return ""
