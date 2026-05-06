from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DocumentMetadata:
    source: str
    ticker: str | None = None
    company: str | None = None
    form_type: str | None = None
    filing_date: str | None = None
    section: str | None = None
    source_url: str | None = None

    def with_section(self, section: str | None) -> DocumentMetadata:
        return replace(self, section=section)


@dataclass(frozen=True)
class Document:
    text: str
    metadata: DocumentMetadata


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    start_char: int
    end_char: int
