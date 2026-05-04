from __future__ import annotations

from retrieval.embeddings import tokenize
from retrieval.vector_store import SearchResult


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
            scored.append((result, overlap + section_boost + (0.15 * result.score)))

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
    return 0.0


def _section_filtered_results(
    query_terms: set[str], results: list[SearchResult]
) -> list[SearchResult]:
    if not ({"risk", "risks"} & query_terms and {"factor", "factors"} & query_terms):
        return results
    risk_section_results = [
        result
        for result in results
        if "risk factors" in (result.chunk.metadata.section or "").lower()
    ]
    return risk_section_results or results
