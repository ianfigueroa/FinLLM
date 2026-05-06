from __future__ import annotations

from pathlib import Path

from ingestion.document_cleaner import clean_text
from ingestion.metadata import Document, DocumentMetadata


def load_fed_statement(
    path: str | Path,
    *,
    title: str,
    statement_date: str,
    source_url: str | None = None,
) -> Document:
    source_path = Path(path)
    return Document(
        text=clean_text(source_path.read_text(encoding="utf-8")),
        metadata=DocumentMetadata(
            ticker="FED",
            company="Federal Reserve",
            form_type="fed-statement",
            filing_date=statement_date,
            section=title,
            source=str(source_path),
            source_url=source_url,
        ),
    )
