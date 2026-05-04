from __future__ import annotations

import re
from dataclasses import dataclass

from retrieval.vector_store import SearchResult

_MARKER_PATTERN = re.compile(r"\[(C\d+)\]")


@dataclass(frozen=True)
class Citation:
    marker: str
    chunk_id: str
    source: str
    section: str | None
    ticker: str | None
    start_char: int
    end_char: int
    snippet: str


@dataclass(frozen=True)
class CitationValidation:
    passed: bool
    used_markers: list[str]
    invented_markers: list[str]
    missing_citations: bool


def build_citations(results: list[SearchResult]) -> list[Citation]:
    citations: list[Citation] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        citations.append(
            Citation(
                marker=f"C{index}",
                chunk_id=chunk.chunk_id,
                source=chunk.metadata.source_url or chunk.metadata.source,
                section=chunk.metadata.section,
                ticker=chunk.metadata.ticker,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                snippet=_snippet(chunk.text),
            )
        )
    return citations


def extract_markers(answer: str) -> list[str]:
    return _MARKER_PATTERN.findall(answer)


def validate_answer_citations(answer: str, citations: list[Citation]) -> CitationValidation:
    used_markers = extract_markers(answer)
    allowed_markers = {citation.marker for citation in citations}
    invented = [marker for marker in used_markers if marker not in allowed_markers]
    missing = len(used_markers) == 0

    return CitationValidation(
        passed=not invented and not missing,
        used_markers=used_markers,
        invented_markers=invented,
        missing_citations=missing,
    )


def _snippet(text: str, max_chars: int = 260) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3].rstrip()}..."
