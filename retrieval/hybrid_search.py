from __future__ import annotations

import math
from collections import Counter

from retrieval.embeddings import tokenize
from retrieval.vector_store import InMemoryVectorStore, SearchResult


class HybridSearch:
    """Combine dense vector search with local BM25-style sparse scoring."""

    def __init__(self, vector_store: InMemoryVectorStore, *, dense_weight: float = 0.55) -> None:
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1")
        self._vector_store = vector_store
        self._dense_weight = dense_weight

    def search(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 8,
    ) -> list[SearchResult]:
        dense_results = {
            result.chunk.chunk_id: result
            for result in self._vector_store.search(query, filters=filters, limit=max(limit * 4, limit))
        }
        sparse_scores = self._sparse_scores(query, filters)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in self._vector_store.all_chunks()}
        candidate_ids = set(dense_results) | set(sparse_scores)

        scored: list[tuple[str, float]] = []
        for chunk_id in candidate_ids:
            dense_score = dense_results.get(chunk_id).score if chunk_id in dense_results else 0.0
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            score = (self._dense_weight * dense_score) + ((1 - self._dense_weight) * sparse_score)
            if score > 0:
                scored.append((chunk_id, score))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
        return [
            SearchResult(chunk=chunk_by_id[chunk_id], score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

    def _sparse_scores(self, query: str, filters: dict[str, str] | None) -> dict[str, float]:
        query_terms = tokenize(query)
        if not query_terms:
            return {}

        chunks = [
            chunk
            for chunk in self._vector_store.all_chunks()
            if all(getattr(chunk.metadata, key, None) == value for key, value in (filters or {}).items())
        ]
        document_frequency: Counter[str] = Counter()
        chunk_terms = {}
        for chunk in chunks:
            terms = tokenize(chunk.text)
            chunk_terms[chunk.chunk_id] = terms
            document_frequency.update(set(terms))

        scores: dict[str, float] = {}
        total_documents = max(len(chunks), 1)
        for chunk in chunks:
            terms = chunk_terms[chunk.chunk_id]
            counts = Counter(terms)
            score = 0.0
            for term in query_terms:
                if counts[term] == 0:
                    continue
                idf = math.log(1 + ((total_documents - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)))
                score += idf * counts[term] / max(len(terms), 1)
            if score > 0:
                scores[chunk.chunk_id] = score
        return _normalize_scores(scores)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    return {key: value / max_score for key, value in scores.items()}
