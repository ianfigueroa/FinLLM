from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, status

from agent.graph import ResearchAgent
from api.schemas import ApiResponse, ChatRequest
from evals.datasets import SAMPLE_EVAL_CASES
from evals.regression_tests import run_regression_cases
from ingestion.chunker import chunk_document
from ingestion.sec_loader import load_sec_filing
from retrieval.embeddings import HashEmbeddingModel
from retrieval.vector_store import InMemoryVectorStore


class AppState:
    def __init__(self) -> None:
        self.store = InMemoryVectorStore(HashEmbeddingModel())


def create_app() -> FastAPI:
    app = FastAPI(title="FinLLM Research Agent", version="0.1.0")
    state = AppState()

    @app.get("/health", response_model=ApiResponse)
    def health() -> ApiResponse:
        return ApiResponse(data={"status": "ok"})

    @app.post(
        "/api/v1/ingestions/sample",
        response_model=ApiResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def ingest_sample() -> ApiResponse:
        sample_path = Path("examples/sample_docs/acme_10k_2025.txt")
        document = load_sec_filing(
            sample_path,
            ticker="ACME",
            company="Acme Corp",
            form_type="10-K",
            filing_date="2025-02-15",
        )
        chunks = chunk_document(document, max_chars=420, overlap_chars=60)
        state.store.upsert(chunks)
        return ApiResponse(data={"documents_ingested": 1, "chunks_indexed": len(chunks)})

    @app.get("/api/v1/ingestions/status", response_model=ApiResponse)
    def ingestion_status() -> ApiResponse:
        chunks = state.store.all_chunks()
        return ApiResponse(data={"chunks_indexed": len(chunks)})

    @app.post("/api/v1/chat", response_model=ApiResponse)
    def chat(request: ChatRequest) -> ApiResponse:
        response = ResearchAgent(state.store, mode=request.mode).answer(
            request.question,
            filters=request.filters,
        )
        return ApiResponse(data=_agent_response_payload(response))

    @app.post("/api/v1/evals", response_model=ApiResponse)
    def run_evals() -> ApiResponse:
        agent = ResearchAgent(state.store, mode="self_verify")
        results = run_regression_cases(
            SAMPLE_EVAL_CASES,
            lambda case: _answer_for_eval(agent, case.question),
        )
        pass_rate = sum(1 for result in results if result["passed"]) / max(len(results), 1)
        return ApiResponse(data={"regression_pass_rate": pass_rate, "cases": results})

    return app


app = create_app()


def _agent_response_payload(response: object) -> dict[str, object]:
    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "limitations": response.limitations,
        "disclaimer": response.disclaimer,
        "mode": response.mode,
        "citations": [citation.__dict__ for citation in response.citations],
        "retrieved_chunks": [
            {
                "chunk_id": result.chunk.chunk_id,
                "text": result.chunk.text,
                "score": result.score,
                "metadata": result.chunk.metadata.__dict__,
            }
            for result in response.retrieved_chunks
        ],
        "verification": response.verification.__dict__,
    }


def _answer_for_eval(agent: ResearchAgent, question: str) -> tuple[str, list[str], list[str]]:
    response = agent.answer(question)
    return (
        response.answer,
        [result.chunk.chunk_id for result in response.retrieved_chunks],
        [citation.marker for citation in response.citations],
    )
