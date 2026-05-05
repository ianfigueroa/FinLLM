from __future__ import annotations

from retrieval.citations import Citation, CitationValidation, validate_answer_citations


class CitationVerifier:
    def verify(self, answer: str, citations: list[Citation]) -> CitationValidation:
        return validate_answer_citations(answer, citations)

