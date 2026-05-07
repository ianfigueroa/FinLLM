from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ingestion.metadata import DocumentChunk
from retrieval.embeddings import EmbeddingModel, cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
    rank: int


class VectorStore(Protocol):
    def upsert(self, chunks: list[DocumentChunk]) -> None: ...

    def search(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 8,
    ) -> list[SearchResult]: ...

    def all_chunks(self) -> list[DocumentChunk]: ...


class InMemoryVectorStore:
    """Small vector store used for tests, local demos, and eval fixtures."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedding_model = embedding_model
        self._records: dict[str, tuple[DocumentChunk, list[float]]] = {}

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._records[chunk.chunk_id] = (chunk, self._embedding_model.embed(chunk.text))

    def search(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 8,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        query_embedding = self._embedding_model.embed(query)
        scored: list[tuple[DocumentChunk, float]] = []

        for chunk, embedding in self._records.values():
            if not _metadata_matches(chunk, filters):
                continue
            score = cosine_similarity(query_embedding, embedding)
            if score > 0:
                scored.append((chunk, score))

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(ranked, start=1)
        ]

    def all_chunks(self) -> list[DocumentChunk]:
        return [chunk for chunk, _ in self._records.values()]


def _metadata_matches(chunk: DocumentChunk, filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    return all(getattr(chunk.metadata, key, None) == expected for key, expected in filters.items())
