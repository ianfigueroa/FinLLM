from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import AgentResponse, ResearchAgent
from agent.memory import ConversationMemory, MemoryTurn
from api.schemas import ApiResponse, ChatRequest, RaftExperimentRequest, SecUrlIngestionRequest
from evals.citation_eval import citation_correctness
from evals.datasets import SAMPLE_EVAL_CASES
from evals.hallucination_eval import hallucination_rate
from evals.ragas_eval import answer_relevance, context_relevance, retrieval_precision
from evals.regression_tests import run_regression_cases
from evals.tool_eval import tool_call_success_rate
from finetuning.evaluate_finetuned import simulate_finetuned_eval
from finetuning.make_lora_dataset import make_lora_records
from finetuning.make_raft_dataset import make_raft_examples
from finetuning.train_lora import simulate_lora_training
from ingestion.chunker import chunk_document
from ingestion.metadata import Document, DocumentMetadata
from ingestion.sec_loader import load_sec_filing
from ingestion.sec_url_loader import load_sec_filing_url
from ingestion.upload_loader import UploadedFileError, extract_upload_text
from observability.structured_logger import StructuredLogger
from retrieval.chroma_store import ChromaVectorStore
from retrieval.embeddings import HashEmbeddingModel
from retrieval.vector_store import InMemoryVectorStore, VectorStore


class AppState:
    def __init__(self) -> None:
        embedding_model = HashEmbeddingModel()
        self.store: VectorStore
        if os.getenv("FINLLM_VECTOR_BACKEND") == "chroma":
            self.store = ChromaVectorStore(
                path=os.getenv("FINLLM_STORAGE_DIR", "storage/chroma"),
                embedding_model=embedding_model,
            )
        else:
            self.store = InMemoryVectorStore(embedding_model)
        self.logger = StructuredLogger("finllm.api")
        self.memory = ConversationMemory()


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
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Upload exceeds 10 MB limit")
        try:
            text = extract_upload_text(
                filename=file.filename,
                content_type=file.content_type,
                content=content,
            )
        except UploadedFileError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

        document = Document(
            text=text,
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
        state.memory.append(
            MemoryTurn(
                question=request.question,
                answer=response.answer,
                mode=request.mode,
                citation_markers=[citation.marker for citation in response.citations],
            )
        )
        return ApiResponse(data=_agent_response_payload(response))

    @app.post("/api/v1/evals", response_model=ApiResponse)
    def run_evals() -> ApiResponse:
        missing_eval_chunks = _missing_eval_chunks(state)
        if missing_eval_chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Index sample before running built-in evals; missing expected chunks: "
                    + ", ".join(missing_eval_chunks)
                ),
            )
        agent = ResearchAgent(state.store, mode="self_verify")
        results = run_regression_cases(
            SAMPLE_EVAL_CASES,
            lambda case: _answer_for_eval(agent, case.question, case.filters),
        )
        pass_rate = sum(1 for result in results if result["passed"]) / max(len(results), 1)
        mode_results = _evaluate_modes(state)
        state.logger.event(
            "eval.completed",
            cases=len(results),
            regression_pass_rate=pass_rate,
            modes=len(mode_results),
        )
        return ApiResponse(
            data={
                "regression_pass_rate": pass_rate,
                "cases": results,
                "mode_results": mode_results,
            }
        )

    @app.post("/api/v1/finetuning/raft", response_model=ApiResponse)
    def run_raft_experiment(request: RaftExperimentRequest) -> ApiResponse:
        chunks = state.store.all_chunks()
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Index documents before generating RAFT data",
            )

        examples = make_raft_examples(
            chunks,
            questions_per_chunk=1,
            distractor_count=request.distractor_count,
        )[: request.max_examples]
        lora_records = make_lora_records(examples)
        training_report = simulate_lora_training(
            records=len(lora_records),
            base_model=request.base_model,
        )
        eval_report = simulate_finetuned_eval(
            model_id=str(training_report["model_id"]),
            eval_cases=len(SAMPLE_EVAL_CASES),
        )
        state.logger.event(
            "finetuning.raft.completed",
            raft_examples=len(examples),
            lora_records=len(lora_records),
            status=training_report["status"],
        )
        return ApiResponse(
            data={
                "raft_examples": len(examples),
                "lora_records": len(lora_records),
                "preview": examples[:3],
                "training_report": training_report,
                "eval_report": eval_report,
            }
        )

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


def _answer_for_eval(
    agent: ResearchAgent, question: str, filters: dict[str, str] | None
) -> tuple[str, list[str], list[str]]:
    response = agent.answer(question, filters=filters)
    return (
        response.answer,
        [result.chunk.chunk_id for result in response.retrieved_chunks],
        [citation.marker for citation in response.citations],
    )


def _evaluate_modes(state: AppState) -> list[dict[str, object]]:
    mode_results: list[dict[str, object]] = []
    for mode in ["basic_rag", "rag_rerank", "self_verify"]:
        latencies: list[float] = []
        retrieval_scores: list[float] = []
        context_scores: list[float] = []
        citation_scores: list[float] = []
        hallucination_scores: list[float] = []
        relevance_scores: list[float] = []
        tool_successes: list[bool] = []

        agent = ResearchAgent(state.store, mode=mode)
        for case in SAMPLE_EVAL_CASES:
            started = perf_counter()
            response = agent.answer(case.question, filters=case.filters)
            latencies.append(round((perf_counter() - started) * 1_000, 4))
            retrieved_ids = [result.chunk.chunk_id for result in response.retrieved_chunks]
            evidence = [result.chunk.text for result in response.retrieved_chunks]
            allowed_markers = [citation.marker for citation in response.citations]

            retrieval_scores.append(retrieval_precision(retrieved_ids, case.expected_chunk_ids))
            context_scores.append(context_relevance(case.question, evidence))
            citation_scores.append(citation_correctness(response.answer, allowed_markers))
            hallucination_scores.append(hallucination_rate(response.answer, evidence))
            relevance_scores.append(answer_relevance(case.question, response.answer))
            tool_successes.extend(call.ok for call in response.tool_calls)

        mode_results.append(
            {
                "mode": mode,
                "retrieval_precision": _average(retrieval_scores),
                "context_relevance": _average(context_scores),
                "citation_correctness": _average(citation_scores),
                "hallucination_rate": _average(hallucination_scores),
                "answer_relevance": _average(relevance_scores),
                "avg_latency_ms": _average(latencies),
                "estimated_cost_usd": 0.0,
                "tool_call_success_rate": tool_call_success_rate(tool_successes),
            }
        )
    return mode_results


def _missing_eval_chunks(state: AppState) -> list[str]:
    indexed_ids = {chunk.chunk_id for chunk in state.store.all_chunks()}
    expected_ids = {
        chunk_id
        for case in SAMPLE_EVAL_CASES
        for chunk_id in case.expected_chunk_ids
    }
    return sorted(expected_ids - indexed_ids)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
