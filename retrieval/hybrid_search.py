from __future__ import annotations

import math
from collections import Counter

from ingestion.metadata import DocumentChunk
from retrieval.embeddings import tokenize
from retrieval.vector_store import SearchResult, VectorStore


class HybridSearch:
    """Combine dense vector search with local BM25-style sparse scoring."""

    def __init__(self, vector_store: VectorStore, *, dense_weight: float = 0.55) -> None:
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
            for result in self._vector_store.search(
                query, filters=filters, limit=max(limit * 4, limit)
            )
        }
        chunks = self._filtered_chunks(filters)
        sparse_scores = self._sparse_scores(query, chunks)
        section_scores = _section_intent_scores(query, chunks)
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        candidate_ids = set(dense_results) | set(sparse_scores) | set(section_scores)

        scored: list[tuple[str, float]] = []
        for chunk_id in candidate_ids:
            dense_result = dense_results.get(chunk_id)
            dense_score = dense_result.score if dense_result else 0.0
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            section_score = section_scores.get(chunk_id, 0.0)
            score = (
                (self._dense_weight * dense_score)
                + ((1 - self._dense_weight) * sparse_score)
                + section_score
            )
            if score > 0:
                scored.append((chunk_id, score))

        ranked = sorted(
            scored,
            key=lambda item: (-item[1], chunk_by_id[item[0]].start_char, item[0]),
        )[:limit]
        return [
            SearchResult(chunk=chunk_by_id[chunk_id], score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

    def _filtered_chunks(self, filters: dict[str, str] | None) -> list[DocumentChunk]:
        return [
            chunk
            for chunk in self._vector_store.all_chunks()
            if all(
                getattr(chunk.metadata, key, None) == value
                for key, value in (filters or {}).items()
            )
        ]

    def _sparse_scores(self, query: str, chunks: list[DocumentChunk]) -> dict[str, float]:
        query_terms = tokenize(query)
        if not query_terms:
            return {}

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
                idf = math.log(
                    1
                    + (
                        (total_documents - document_frequency[term] + 0.5)
                        / (document_frequency[term] + 0.5)
                    )
                )
                score += idf * counts[term] / max(len(terms), 1)
            if score > 0:
                scores[chunk.chunk_id] = score
        return _normalize_scores(scores)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    return {key: value / max_score for key, value in scores.items()}


def _section_intent_scores(query: str, chunks: list[DocumentChunk]) -> dict[str, float]:
    query_terms = set(tokenize(query))
    if {"risk", "risks"} & query_terms and {"factor", "factors"} & query_terms:
        return {
            chunk.chunk_id: 1.0
            for chunk in chunks
            if "risk factors" in (chunk.metadata.section or "").lower()
        }
    if _is_mdna_query(query_terms):
        return {
            chunk.chunk_id: 0.8
            for chunk in chunks
            if _is_mdna_section((chunk.metadata.section or "").lower())
        }
    return {}


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
