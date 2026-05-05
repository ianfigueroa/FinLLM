from __future__ import annotations

from evals.ragas_eval import retrieval_precision

__all__ = [
    "mean_reciprocal_rank",
    "retrieval_metrics",
    "retrieval_precision",
    "retrieval_recall_at_k",
]


def retrieval_recall_at_k(
    retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], *, k: int
) -> float:
    if not expected_chunk_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_chunk_ids[:k])
    expected = set(expected_chunk_ids)
    return round(len(top_k & expected) / len(expected), 4)


def mean_reciprocal_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    if not expected_chunk_ids:
        return 0.0
    expected = set(expected_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return round(1.0 / rank, 4)
    return 0.0


def retrieval_metrics(
    retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], *, k: int = 5
) -> dict[str, float]:
    return {
        "precision": retrieval_precision(retrieved_chunk_ids, expected_chunk_ids),
        "recall_at_k": retrieval_recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k=k),
        "mrr": mean_reciprocal_rank(retrieved_chunk_ids, expected_chunk_ids),
    }
