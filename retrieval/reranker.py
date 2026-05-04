from __future__ import annotations

from retrieval.embeddings import tokenize
from retrieval.vector_store import SearchResult


class LexicalReranker:
    """Rerank retrieved chunks by query term coverage and original score."""

    def rerank(self, query: str, results: list[SearchResult], *, limit: int | None = None) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return results[:limit]

        scored: list[tuple[SearchResult, float]] = []
        for result in results:
            chunk_terms = set(tokenize(result.chunk.text))
            overlap = len(query_terms & chunk_terms) / len(query_terms)
            scored.append((result, overlap + (0.15 * result.score)))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        trimmed = ranked[:limit] if limit is not None else ranked
        return [
            SearchResult(chunk=result.chunk, score=score, rank=index)
            for index, (result, score) in enumerate(trimmed, start=1)
        ]

