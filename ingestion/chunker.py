from __future__ import annotations

import re

from ingestion.metadata import Document, DocumentChunk

_SECTION_PATTERN = re.compile(r"^(Item\s+\d+[A-Z]?\.\s+.+)$", re.IGNORECASE)


def chunk_document(
    document: Document,
    *,
    max_chars: int = 1_200,
    overlap_chars: int = 160,
) -> list[DocumentChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars")

    text = document.text.strip()
    if not text:
        return []

    chunks: list[DocumentChunk] = []
    section = document.metadata.section
    start = 0
    index = 0

    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1

        chunk_text = text[start:end].strip()
        section = _section_for_chunk(chunk_text, section)
        chunks.append(
            DocumentChunk(
                chunk_id=_chunk_id(document, index),
                text=chunk_text,
                metadata=document.metadata.with_section(section),
                start_char=start,
                end_char=end,
            )
        )
        index += 1
        if end == len(text):
            break
        next_start = _align_start_to_word_boundary(text, max(0, end - overlap_chars))
        start = next_start if next_start > start else end

    return chunks


def _section_for_chunk(text: str, fallback: str | None) -> str | None:
    for line in text.splitlines():
        match = _SECTION_PATTERN.match(line.strip())
        if match:
            return match.group(1)
    return fallback


def _align_start_to_word_boundary(text: str, start: int) -> int:
    if start <= 0 or start >= len(text):
        return start
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    return start


def _chunk_id(document: Document, index: int) -> str:
    metadata = document.metadata
    parts = [
        metadata.ticker or "UNKNOWN",
        metadata.form_type or "DOC",
        metadata.filing_date or "undated",
        f"{index:04d}",
    ]
    return "-".join(part.replace(" ", "_") for part in parts)
