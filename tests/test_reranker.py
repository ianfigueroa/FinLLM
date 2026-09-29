from __future__ import annotations

from ingestion.metadata import DocumentChunk, DocumentMetadata
from retrieval.reranker import LexicalReranker
from retrieval.vector_store import SearchResult

QUESTION = "What was Apple's gross profit in the latest fiscal year?"


def _result(chunk_id: str, text: str, score: float, section: str | None = None) -> SearchResult:
    chunk = DocumentChunk(
        chunk_id, text, DocumentMetadata(source="10-K", ticker="AAPL", section=section), 0, 0
    )
    return SearchResult(chunk=chunk, score=score, rank=0)


def _ids(results: list[SearchResult]) -> list[str]:
    return [result.chunk.chunk_id for result in results]


def test_figure_table_is_not_demoted_below_prose() -> None:
    table = _result(
        "table",
        "Gross profit 2025 195201000000 2024 180683000000 2023 169148000000\n"
        "Operating income 2025 133050000000 2024 123216000000",
        score=0.5,
    )
    prose = _result("prose", "Apple reports gross margin trends in its annual filing.", 0.5)

    assert _ids(LexicalReranker().rerank(QUESTION, [prose, table]))[0] == "table"


def test_question_stopwords_do_not_outrank_content_terms() -> None:
    # Before stopwords were dropped this scored 5/10 query terms against the answer's 2/10.
    boilerplate = _result("boilerplate", "What was in the fiscal report?", 0.45)
    answer = _result("answer", "Gross profit rose.", 0.45)

    assert _ids(LexicalReranker().rerank(QUESTION, [boilerplate, answer]))[0] == "answer"


def test_risk_factor_questions_still_prefer_risk_section() -> None:
    risk = _result("risk", "Supply constraints could reduce revenue.", 0.2, "Item 1A. Risk Factors")
    other = _result("other", "Supply constraints were discussed.", 0.6, "Item 1. Business")

    ranked = LexicalReranker().rerank("What risk factors relate to supply?", [other, risk])

    assert _ids(ranked) == ["risk"]
