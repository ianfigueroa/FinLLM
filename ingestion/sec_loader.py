from __future__ import annotations

from pathlib import Path

from ingestion.document_cleaner import clean_text
from ingestion.metadata import Document, DocumentMetadata


def load_sec_filing(
    path: str | Path,
    *,
    ticker: str,
    company: str,
    form_type: str,
    filing_date: str,
    source_url: str | None = None,
) -> Document:
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    return Document(
        text=clean_text(text),
        metadata=DocumentMetadata(
            ticker=ticker.upper(),
            company=company,
            form_type=form_type.upper(),
            filing_date=filing_date,
            source=str(source_path),
            source_url=source_url,
        ),
    )

