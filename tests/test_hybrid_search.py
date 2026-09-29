from __future__ import annotations

from ingestion.metadata import DocumentChunk, DocumentMetadata
from retrieval.hybrid_search import HybridSearch
from retrieval.vector_store import SearchResult


class _FixedDenseStore:
    """Dense side returns fixed scores in the narrow band neural embedders produce."""

    def __init__(self, chunks: list[DocumentChunk], dense: dict[str, float]) -> None:
        self._chunks = chunks
        self._dense = dense

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(
        self, query: str, *, filters: dict[str, str] | None = None, limit: int = 8
    ) -> list[SearchResult]:
        by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        ranked = sorted(self._dense.items(), key=lambda item: -item[1])[:limit]
        return [
            SearchResult(chunk=by_id[chunk_id], score=score, rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

    def all_chunks(self) -> list[DocumentChunk]:
        return self._chunks


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id, text, DocumentMetadata(source="10-K"), 0, len(text))


def test_weakest_dense_candidate_does_not_outrank_a_strong_sparse_match() -> None:
    chunks = [
        _chunk("dense-best", "Our products are sold worldwide."),
        _chunk("dense-weakest", "We compete in many markets."),
        _chunk("sparse-match", "Goodwill 119,651 119,509"),
    ]
    store = _FixedDenseStore(chunks, {"dense-best": 0.90, "dense-weakest": 0.86})

    ids = [r.chunk.chunk_id for r in HybridSearch(store).search("goodwill balance", limit=3)]

    assert "dense-weakest" not in ids[: ids.index("sparse-match")]
