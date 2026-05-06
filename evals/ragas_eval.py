from __future__ import annotations

from retrieval.embeddings import tokenize


def retrieval_precision(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str]) -> float:
    if not expected_chunk_ids:
        return 0.0
    retrieved = set(retrieved_chunk_ids)
    expected = set(expected_chunk_ids)
    return round(len(retrieved & expected) / len(expected), 4)


def context_relevance(question: str, contexts: list[str]) -> float:
    return _term_overlap(question, " ".join(contexts))


def answer_relevance(question: str, answer: str) -> float:
    return _term_overlap(question, answer)


def _term_overlap(left: str, right: str) -> float:
    left_terms = set(tokenize(left))
    if not left_terms:
        return 0.0
    right_terms = set(tokenize(right))
    return round(len(left_terms & right_terms) / len(left_terms), 4)
