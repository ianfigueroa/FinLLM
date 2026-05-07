from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import AgentResponse, ResearchAgent
from api.schemas import ApiResponse, ChatRequest, SecUrlIngestionRequest
from evals.datasets import SAMPLE_EVAL_CASES
from evals.regression_tests import run_regression_cases
from ingestion.chunker import chunk_document
from ingestion.document_cleaner import clean_text
from ingestion.metadata import Document, DocumentMetadata
from ingestion.sec_loader import load_sec_filing
from ingestion.sec_url_loader import load_sec_filing_url
from observability.structured_logger import StructuredLogger
from retrieval.embeddings import HashEmbeddingModel
from retrieval.vector_store import InMemoryVectorStore


class AppState:
    def __init__(self) -> None:
        self.store = InMemoryVectorStore(HashEmbeddingModel())
        self.logger = StructuredLogger("finllm.api")


def create_app() -> FastAPI:
    app = FastAPI(title="FinLLM Research Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5180",
            "http://localhost:5180",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:5174",
            "http://localhost:5174",
            "http://127.0.0.1:5175",
            "http://localhost:5175",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
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
        ticker: Annotated[str, Form(min_length=1, max_length=12)],
        company: Annotated[str, Form(min_length=1, max_length=120)],
        form_type: Annotated[str, Form(min_length=1, max_length=32)],
        filing_date: Annotated[str, Form(min_length=4, max_length=32)],
        file: Annotated[UploadFile, File()],
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

    @app.post(
        "/api/v1/ingestions/sec-url",
        response_model=ApiResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def ingest_sec_url(request: SecUrlIngestionRequest) -> ApiResponse:
        try:
            document = load_sec_filing_url(
                request.url,
                ticker=request.ticker,
                company=request.company,
                form_type=request.form_type,
                filing_date=request.filing_date,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="SEC filing fetch failed") from exc

        chunks = chunk_document(document, max_chars=420, overlap_chars=60)
        state.store.upsert(chunks)
        ticker = request.ticker.upper()
        state.logger.event(
            "ingestion.sec_url.completed",
            documents=1,
            chunks=len(chunks),
            ticker=ticker,
            source_url=document.metadata.source_url,
        )
        return ApiResponse(
            data={
                "documents_ingested": 1,
                "chunks_indexed": len(chunks),
                "ticker": ticker,
                "source_url": document.metadata.source_url,
            }
        )

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
            tool_calls=len(response.tool_calls),
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


def _agent_response_payload(response: AgentResponse) -> dict[str, object]:
    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "limitations": response.limitations,
        "disclaimer": response.disclaimer,
        "mode": response.mode,
        "citations": [citation.__dict__ for citation in response.citations],
        "tool_calls": [tool_call.__dict__ for tool_call in response.tool_calls],
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
