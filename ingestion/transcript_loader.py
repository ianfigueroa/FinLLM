from __future__ import annotations

from pathlib import Path

from ingestion.document_cleaner import clean_text
from ingestion.metadata import Document, DocumentMetadata


def load_transcript(
    path: str | Path,
    *,
    ticker: str,
    company: str,
    event_date: str,
    source_url: str | None = None,
) -> Document:
    source_path = Path(path)
    return Document(
        text=clean_text(source_path.read_text(encoding="utf-8")),
        metadata=DocumentMetadata(
            ticker=ticker.upper(),
            company=company,
            form_type="earnings-call",
            filing_date=event_date,
            source=str(source_path),
            source_url=source_url,
        ),
    )

