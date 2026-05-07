from __future__ import annotations

import os
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ingestion.document_cleaner import clean_text
from ingestion.metadata import Document, DocumentMetadata

_SEC_HOSTS = {"sec.gov", "www.sec.gov"}
_SEC_ARCHIVE_PREFIX = "/Archives/edgar/data/"
_SEC_DOCUMENT_SUFFIXES = (".htm", ".html", ".txt")
_MAX_FILING_BYTES = 10 * 1024 * 1024


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


def load_sec_filing_from_html(
    html: str,
    *,
    source_url: str,
    ticker: str,
    company: str,
    form_type: str,
    filing_date: str,
) -> Document:
    normalized_url = normalize_sec_filing_url(source_url)
    return Document(
        text=extract_sec_html_text(html),
        metadata=DocumentMetadata(
            source=normalized_url,
            ticker=ticker.upper(),
            company=company,
            form_type=form_type.upper(),
            filing_date=filing_date,
            source_url=normalized_url,
        ),
    )


def load_sec_filing_url(
    url: str,
    *,
    ticker: str,
    company: str,
    form_type: str,
    filing_date: str,
) -> Document:
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
    return load_sec_filing_from_html(
        response.text,
        source_url=normalized_url,
        ticker=ticker,
        company=company,
        form_type=form_type,
        filing_date=filing_date,
    )
