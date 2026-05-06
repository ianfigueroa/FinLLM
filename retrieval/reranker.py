from __future__ import annotations

import re

from retrieval.embeddings import tokenize
from retrieval.vector_store import SearchResult

_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")


class LexicalReranker:
    """Rerank retrieved chunks by query term coverage and original score."""

    def rerank(
        self, query: str, results: list[SearchResult], *, limit: int | None = None
    ) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return results[:limit]

        results = _section_filtered_results(query_terms, results)
        scored: list[tuple[SearchResult, float]] = []
        for result in results:
            chunk_terms = set(tokenize(result.chunk.text))
            overlap = len(query_terms & chunk_terms) / len(query_terms)
            section_boost = _section_boost(query_terms, result)
            table_penalty = _table_penalty(result.chunk.text)
            scored.append((result, overlap + section_boost + table_penalty + (0.15 * result.score)))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        trimmed = ranked[:limit] if limit is not None else ranked
        return [
            SearchResult(chunk=result.chunk, score=score, rank=index)
            for index, (result, score) in enumerate(trimmed, start=1)
        ]


def _section_boost(query_terms: set[str], result: SearchResult) -> float:
    section = (result.chunk.metadata.section or "").lower()
    if {"risk", "risks"} & query_terms and "risk factors" in section:
        return 0.45
    if _is_mdna_query(query_terms) and _is_mdna_section(section):
        return 0.55
    if _is_mdna_query(query_terms) and "financial statements" in section:
        return -0.2
    return 0.0


def _section_filtered_results(
    query_terms: set[str], results: list[SearchResult]
) -> list[SearchResult]:
    if {"risk", "risks"} & query_terms and {"factor", "factors"} & query_terms:
        risk_section_results = [
            result
            for result in results
            if "risk factors" in (result.chunk.metadata.section or "").lower()
        ]
        return risk_section_results or results

    if _is_mdna_query(query_terms):
        mdna_results = [
            result
            for result in results
            if _is_mdna_section((result.chunk.metadata.section or "").lower())
        ]
        return mdna_results or results

    return results


def _is_mdna_query(query_terms: set[str]) -> bool:
    return bool(
        {"revenue", "revenues", "margin", "margins", "driver", "drivers", "growth"}
        & query_terms
    )


def _is_mdna_section(section: str) -> bool:
    return (
        ("management" in section and "discussion" in section)
        or "results of operations" in section
        or "liquidity and capital resources" in section
    )


def _table_penalty(text: str) -> float:
    terms = tokenize(text)
    if not terms:
        return 0.0
    numeric_ratio = len(_NUMBER_PATTERN.findall(text)) / max(len(terms), 1)
    return -0.35 if numeric_ratio > 0.25 else 0.0
