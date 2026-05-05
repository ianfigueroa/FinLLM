from __future__ import annotations

from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status

from agent.graph import ResearchAgent
from api.schemas import ApiResponse, ChatRequest
from evals.datasets import SAMPLE_EVAL_CASES
from evals.regression_tests import run_regression_cases
from ingestion.chunker import chunk_document
from ingestion.document_cleaner import clean_text
from ingestion.metadata import Document, DocumentMetadata
from ingestion.sec_loader import load_sec_filing
from observability.structured_logger import StructuredLogger
from retrieval.embeddings import HashEmbeddingModel
from retrieval.vector_store import InMemoryVectorStore


class AppState:
    def __init__(self) -> None:
        self.store = InMemoryVectorStore(HashEmbeddingModel())
        self.logger = StructuredLogger("finllm.api")


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
        state.logger.event("ingestion.sample.completed", documents=1, chunks=len(chunks))
        return ApiResponse(data={"documents_ingested": 1, "chunks_indexed": len(chunks)})

    @app.post(
        "/api/v1/documents/upload",
        response_model=ApiResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_document(
        ticker: str = Form(..., min_length=1, max_length=12),
        company: str = Form(..., min_length=1, max_length=120),
        form_type: str = Form(..., min_length=1, max_length=32),
        filing_date: str = Form(..., min_length=4, max_length=32),
        file: UploadFile = File(...),
    ) -> ApiResponse:
        if file.content_type not in {"text/plain", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="Only plain text uploads are supported")
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload exceeds 10 MB limit")
        document = Document(
            text=clean_text(content.decode("utf-8")),
            metadata=DocumentMetadata(
                ticker=ticker.upper(),
                company=company,
                form_type=form_type.upper(),
                filing_date=filing_date,
                source=file.filename or "uploaded.txt",
            ),
        )
        chunks = chunk_document(document, max_chars=420, overlap_chars=60)
        state.store.upsert(chunks)
        state.logger.event(
            "ingestion.upload.completed",
            documents=1,
            chunks=len(chunks),
            ticker=ticker.upper(),
        )
        return ApiResponse(data={"documents_ingested": 1, "chunks_indexed": len(chunks)})

    @app.get("/api/v1/ingestions/status", response_model=ApiResponse)
    def ingestion_status() -> ApiResponse:
        chunks = state.store.all_chunks()
        return ApiResponse(data={"chunks_indexed": len(chunks)})

    @app.post("/api/v1/chat", response_model=ApiResponse)
    def chat(request: ChatRequest) -> ApiResponse:
        started = perf_counter()
        response = ResearchAgent(state.store, mode=request.mode).answer(
            request.question,
            filters=request.filters,
        )
        latency_ms = round((perf_counter() - started) * 1_000, 4)
        state.logger.event(
            "chat.completed",
            mode=request.mode,
            retrieved_chunks=len(response.retrieved_chunks),
            citations=len(response.citations),
            latency_ms=latency_ms,
            estimated_cost_usd=0.0,
            citation_verification_passed=response.verification.passed,
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
        state.logger.event("eval.completed", cases=len(results), regression_pass_rate=pass_rate)
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
