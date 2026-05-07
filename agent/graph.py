from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent.planner import Planner
from agent.tools import BacktestTool, CalculatorTool, MarketDataTool, SQLMetadataTool, ToolResult
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
    tool_calls: list[ToolCallRecord]
    confidence: float
    limitations: list[str]
    disclaimer: str
    mode: str
    verification: CitationValidation


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    input: dict[str, Any]
    ok: bool
    output: Any = None
    error: str = ""


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
            if plan.tool_required:
                return self._tool_only_response(question, filters)
            return self._insufficient_response("Retrieval was not requested for this question.")

        results = self._retrieve(question, filters=filters, limit=limit)
        if not results:
            return self._insufficient_response("No retrieved evidence supported the question.")

        citations = build_citations(results)
        tool_calls = self._run_tools(question, filters)
        answer = self._draft_grounded_answer(question, results, citations, tool_calls)
        verification = self._verifier.verify(answer, citations)
        confidence = _confidence(results, verification)

        return AgentResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=results,
            tool_calls=tool_calls,
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
        tool_calls: list[ToolCallRecord],
    ) -> str:
        query_terms = set(tokenize(question)) - {"what", "does", "did", "the", "about", "mention"}
        lines = ["Facts from retrieved evidence:"]
        for result, citation in zip(results, citations, strict=True):
            sentence = _best_sentence(result.chunk.text, query_terms)
            lines.append(f"- {sentence} [{citation.marker}]")
        if tool_calls:
            lines.append("Tool analysis:")
            for call in tool_calls:
                if call.ok:
                    lines.append(f"- {call.name}: {call.output}")
                else:
                    lines.append(f"- {call.name}: unavailable ({call.error})")
        lines.append("Inference: Limited to the cited retrieved evidence.")
        return "\n".join(lines)

    def _run_tools(
        self,
        question: str,
        filters: dict[str, str] | None,
    ) -> list[ToolCallRecord]:
        lowered = question.lower()
        calls: list[ToolCallRecord] = []

        expression = _extract_arithmetic_expression(question)
        if expression:
            calls.append(_call_tool(CalculatorTool(), {"expression": expression}))

        if "backtest" in lowered:
            ticker = _ticker_from_question_or_filters(question, filters)
            prices_call = _call_tool(MarketDataTool(_local_market_data()), {"ticker": ticker})
            calls.append(prices_call)
            if prices_call.ok:
                calls.append(
                    _call_tool(
                        BacktestTool(),
                        {
                            "prices": prices_call.output["prices"],
                            "signals": _sample_signals(prices_call.output["prices"]),
                            "threshold": 0.5,
                        },
                    )
                )

        if "metadata" in lowered or "indexed documents" in lowered:
            rows = [chunk.metadata.__dict__ for chunk in self._vector_store.all_chunks()]
            calls.append(
                _call_tool(
                    SQLMetadataTool(rows),
                    {
                        "query": (
                            "SELECT DISTINCT ticker, company, form_type, filing_date "
                            "FROM documents ORDER BY ticker"
                        )
                    },
                )
            )

        return calls

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
            tool_calls=[],
            confidence=0.0,
            limitations=["No supporting chunks were retrieved."],
            disclaimer="Research analysis only; not financial advice.",
            mode=self._mode,
            verification=validation,
        )

    def _tool_only_response(
        self,
        question: str,
        filters: dict[str, str] | None,
    ) -> AgentResponse:
        tool_calls = self._run_tools(question, filters)
        if not tool_calls:
            return self._insufficient_response("No tool could handle the question.")
        lines = ["Tool analysis:"]
        for call in tool_calls:
            if call.ok:
                lines.append(f"- {call.name}: {call.output}")
            else:
                lines.append(f"- {call.name}: unavailable ({call.error})")
        lines.append(
            "Inference: Limited to the local tool output; no document evidence was retrieved."
        )
        validation = CitationValidation(
            passed=all(call.ok for call in tool_calls),
            used_markers=[],
            invented_markers=[],
            missing_citations=False,
        )
        return AgentResponse(
            answer="\n".join(lines),
            citations=[],
            retrieved_chunks=[],
            tool_calls=tool_calls,
            confidence=0.8 if validation.passed else 0.2,
            limitations=["Tool-only response; no document citations were required."],
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


def _call_tool(tool: Any, payload: dict[str, Any]) -> ToolCallRecord:
    result: ToolResult = tool.run(payload)
    return ToolCallRecord(
        name=tool.name,
        input=payload,
        ok=result.ok,
        output=result.output,
        error=result.error,
    )


def _extract_arithmetic_expression(question: str) -> str | None:
    if "calculate" not in question.lower():
        return None
    match = re.search(r"calculate\s+([0-9][0-9\s+\-*/().]*[0-9)])", question, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _ticker_from_question_or_filters(question: str, filters: dict[str, str] | None) -> str:
    if filters and filters.get("ticker"):
        return filters["ticker"].upper()
    tickers = re.findall(r"\b[A-Z]{2,5}\b", question)
    return tickers[0] if tickers else "ACME"


def _local_market_data() -> dict[str, list[dict[str, float | str]]]:
    return {
        "ACME": [
            {"date": "2025-01-01", "close": 100.0},
            {"date": "2025-01-02", "close": 106.0},
            {"date": "2025-01-03", "close": 103.0},
            {"date": "2025-01-04", "close": 111.0},
        ],
        "NVDA": [
            {"date": "2025-01-01", "close": 140.0},
            {"date": "2025-01-02", "close": 146.0},
            {"date": "2025-01-03", "close": 142.0},
            {"date": "2025-01-04", "close": 151.0},
        ],
    }


def _sample_signals(prices: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    return [
        {"date": price["date"], "score": 0.8 if index % 2 == 0 else 0.2}
        for index, price in enumerate(prices)
    ]
