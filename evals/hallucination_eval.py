from __future__ import annotations

from retrieval.embeddings import tokenize

_STOPWORDS = {"the", "a", "an", "and", "or", "because", "from", "with", "to", "of", "in"}


def hallucination_rate(answer: str, evidence: list[str]) -> float:
    answer_terms = set(tokenize(answer)) - _STOPWORDS
    if not answer_terms:
        return 0.0
    evidence_terms = set(tokenize(" ".join(evidence)))
    unsupported = answer_terms - evidence_terms
    return round(len(unsupported) / len(answer_terms), 4)
