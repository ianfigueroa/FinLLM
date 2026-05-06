from __future__ import annotations

from dataclasses import dataclass

from agent.planner import Planner
from agent.prompts import ANSWER_POLICY
from agent.verifier import CitationVerifier
from retrieval.citations import Citation, CitationValidation, build_citations
from retrieval.embeddings import tokenize
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import LexicalReranker
from retrieval.vector_store import InMemoryVectorStore, SearchResult


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[SearchResult]
    confidence: float
    limitations: list[str]
    disclaimer: str
    mode: str
    verification: CitationValidation


class ResearchAgent:
    def __init__(self, vector_store: InMemoryVectorStore, *, mode: str = "basic_rag") -> None:
        self._vector_store = vector_store
        self._mode = mode
        self._planner = Planner()
        self._verifier = CitationVerifier()
        self._reranker = LexicalReranker()

    def answer(
        self,
        question: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 5,
    ) -> AgentResponse:
        plan = self._planner.plan(question)
        if not plan.retrieval_required:
            return self._insufficient_response("Retrieval was not requested for this question.")

        results = self._retrieve(question, filters=filters, limit=limit)
        if not results:
            return self._insufficient_response("No retrieved evidence supported the question.")

        citations = build_citations(results)
        answer = self._draft_grounded_answer(question, results, citations)
        verification = self._verifier.verify(answer, citations)
        confidence = _confidence(results, verification)

        return AgentResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=results,
            confidence=confidence,
            limitations=_limitations(self._mode),
            disclaimer="Research analysis only; not financial advice.",
            mode=self._mode,
            verification=verification,
        )

    def _retrieve(
        self,
        question: str,
        *,
        filters: dict[str, str] | None,
        limit: int,
    ) -> list[SearchResult]:
        if self._mode == "basic_rag":
            return self._vector_store.search(question, filters=filters, limit=limit)
        results = HybridSearch(self._vector_store).search(
            question, filters=filters, limit=max(limit * 2, limit)
        )
        return self._reranker.rerank(question, results, limit=limit)

    def _draft_grounded_answer(
        self,
        question: str,
        results: list[SearchResult],
        citations: list[Citation],
    ) -> str:
        query_terms = set(tokenize(question)) - {"what", "does", "did", "the", "about", "mention"}
        lines = ["Facts from retrieved evidence:"]
        for result, citation in zip(results, citations, strict=True):
            sentence = _best_sentence(result.chunk.text, query_terms)
            lines.append(f"- {sentence} [{citation.marker}]")
        lines.append("Inference: Limited to the cited retrieved evidence.")
        lines.append(f"Policy: {ANSWER_POLICY}")
        return "\n".join(lines)

    def _insufficient_response(self, reason: str) -> AgentResponse:
        validation = CitationValidation(
            passed=False,
            used_markers=[],
            invented_markers=[],
            missing_citations=True,
        )
        return AgentResponse(
            answer=f"Evidence is insufficient to answer from the indexed documents. {reason}",
            citations=[],
            retrieved_chunks=[],
            confidence=0.0,
            limitations=["No supporting chunks were retrieved."],
            disclaimer="Research analysis only; not financial advice.",
            mode=self._mode,
            verification=validation,
        )


def _best_sentence(text: str, query_terms: set[str]) -> str:
    sentences = [
        sentence.strip() for sentence in text.replace("\n", " ").split(".") if sentence.strip()
    ]
    if not sentences:
        return "Retrieved chunk contains no sentence-like evidence."
    return max(sentences, key=lambda sentence: len(set(tokenize(sentence)) & query_terms))


def _confidence(results: list[SearchResult], verification: CitationValidation) -> float:
    if not verification.passed:
        return 0.0
    score = min(0.95, 0.45 + (0.12 * len(results)) + max(result.score for result in results))
    return round(score, 2)


def _limitations(mode: str) -> list[str]:
    limitations = ["Answers are only as complete as the indexed corpus and retrieved chunks."]
    if mode == "self_verify":
        limitations.append("Self-verification checks citation format, not economic truth.")
    return limitations
