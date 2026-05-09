from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from agent.graph import AgentResponse, ResearchAgent
from agent.llm import build_answer_generator, load_llm_settings
from agent.memory import ConversationMemory, MemoryTurn
from api.schemas import (
    ApiResponse,
    ChatRequest,
    RaftExperimentRequest,
    SecUrlIngestionRequest,
    SecUrlMetadataRequest,
    ThreeStatementModelRequest,
)
from evals.citation_eval import citation_correctness
from evals.datasets import SAMPLE_EVAL_CASES
from evals.hallucination_eval import hallucination_rate
from evals.ragas_eval import answer_relevance, context_relevance, retrieval_precision
from evals.regression_cases import run_regression_cases
from evals.retrieval_eval import mean_reciprocal_rank, retrieval_recall_at_k
from evals.run_store import EvalRunStore
from evals.tool_eval import tool_call_success_rate
from finetuning.evaluate_finetuned import simulate_finetuned_eval
from finetuning.make_lora_dataset import make_lora_jsonl, make_lora_records
from finetuning.make_raft_dataset import make_raft_examples
from finetuning.train_lora import simulate_lora_training
from ingestion.chunker import chunk_document
from ingestion.metadata import Document, DocumentChunk, DocumentMetadata
from ingestion.sec_loader import load_sec_filing
from ingestion.sec_url_loader import (
    SecFilingMetadata,
    fetch_sec_filing_metadata,
    load_sec_filing_url,
)
from ingestion.upload_loader import UploadedFileError, extract_upload_text
from modeling.three_statement import build_three_statement_model
from observability.structured_logger import StructuredLogger
from retrieval.chroma_store import ChromaVectorStore
from retrieval.embeddings import build_embedding_model, load_embedding_settings
from retrieval.vector_store import InMemoryVectorStore, VectorStore

DocumentInventoryKey = tuple[str | None, str | None, str | None, str | None, str]


class AppState:
    def __init__(self) -> None:
        self.embedding_settings = load_embedding_settings()
        embedding_model = build_embedding_model(self.embedding_settings)
        self.store: VectorStore
        self.vector_backend = os.getenv("FINLLM_VECTOR_BACKEND", "memory")
        self.storage_dir = Path(os.getenv("FINLLM_STORAGE_DIR", "storage"))
        self.eval_runs = EvalRunStore(self.storage_dir / "eval_runs.jsonl")
        if os.getenv("FINLLM_VECTOR_BACKEND") == "chroma":
            self.store = ChromaVectorStore(
                path=str(self.storage_dir / "chroma"),
                embedding_model=embedding_model,
            )
        else:
            self.store = InMemoryVectorStore(embedding_model)
        self.llm_settings = load_llm_settings()
        self.answer_generator = build_answer_generator(self.llm_settings)
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
            "http://127.0.0.1:5190",
            "http://localhost:5190",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    state = AppState()

    @app.get("/health", response_model=ApiResponse)
    def health() -> ApiResponse:
        return ApiResponse(data={"status": "ok"})

    @app.get("/api/v1/system/status", response_model=ApiResponse)
    def system_status() -> ApiResponse:
        return ApiResponse(
            data={
                "vector_backend": state.vector_backend,
                "embedding": state.embedding_settings.public_dict(),
                "llm": state.llm_settings.public_dict(),
                "retrieval_modes": ["basic_rag", "rag_rerank", "self_verify"],
                "eval_metrics": [
                    "retrieval_precision",
                    "retrieval_recall_at_5",
                    "retrieval_mrr",
                    "context_relevance",
                    "citation_correctness",
                    "hallucination_rate",
                    "answer_relevance",
                    "avg_latency_ms",
                    "estimated_cost_usd",
                    "tool_call_success_rate",
                    "quality_score",
                ],
            }
        )

    @app.post(
        "/api/v1/ingestions/sample",
        response_model=ApiResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def ingest_sample() -> ApiResponse:
        sample_path = Path("examples/sample_docs/nvda_10k_2026.txt")
        document = load_sec_filing(
            sample_path,
            ticker="NVDA",
            company="NVIDIA Corporation",
            form_type="10-K",
            filing_date="2026-02-21",
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
        ticker = (document.metadata.ticker or request.ticker).upper()
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
                "company": document.metadata.company,
                "form_type": document.metadata.form_type,
                "filing_date": document.metadata.filing_date,
                "source_url": document.metadata.source_url,
            }
        )

    @app.post("/api/v1/ingestions/sec-url/metadata", response_model=ApiResponse)
    def sec_url_metadata(request: SecUrlMetadataRequest) -> ApiResponse:
        try:
            metadata = fetch_sec_filing_metadata(request.url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="SEC filing fetch failed") from exc
        return ApiResponse(data=_sec_metadata_payload(metadata))

    @app.get("/api/v1/ingestions/status", response_model=ApiResponse)
    def ingestion_status() -> ApiResponse:
        chunks = state.store.all_chunks()
        return ApiResponse(
            data={
                "chunks_indexed": len(chunks),
                "documents": _indexed_documents(chunks),
            }
        )

    @app.post("/api/v1/chat", response_model=ApiResponse)
    def chat(request: ChatRequest) -> ApiResponse:
        started = perf_counter()
        filters = request.filters or _infer_single_company_filter(request.question, state)
        response = ResearchAgent(
            state.store,
            mode=request.mode,
            answer_generator=state.answer_generator,
        ).answer(
            request.question,
            filters=filters,
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
        payload = {
            "regression_pass_rate": pass_rate,
            "cases": results,
            "mode_results": mode_results,
            "best_mode": _best_mode(mode_results),
        }
        persisted = state.eval_runs.append(payload)
        state.logger.event(
            "eval.completed",
            cases=len(results),
            regression_pass_rate=pass_rate,
            modes=len(mode_results),
            run_id=persisted["run_id"],
        )
        return ApiResponse(data={**payload, "run_id": persisted["run_id"]})

    @app.get("/api/v1/evals/history", response_model=ApiResponse)
    def eval_history() -> ApiResponse:
        return ApiResponse(
            data={
                "summary": state.eval_runs.summary(),
                "runs": state.eval_runs.recent(limit=20),
            }
        )

    @app.post("/api/v1/models/three-statement", response_model=ApiResponse)
    def build_model(request: ThreeStatementModelRequest) -> ApiResponse:
        ticker = request.ticker.upper()
        chunks = [
            chunk
            for chunk in state.store.all_chunks()
            if (chunk.metadata.ticker or "").upper() == ticker
        ]
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Index documents for this ticker before building a model",
            )

        model = build_three_statement_model(
            chunks,
            ticker=ticker,
            projection_years=request.projection_years,
            revenue_growth=request.revenue_growth,
        )
        state.logger.event(
            "model.three_statement.completed",
            ticker=ticker,
            extracted_sources=len(model.sources),
            projection_years=request.projection_years,
            confidence=model.confidence,
        )
        return ApiResponse(data=model.to_dict())

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

    @app.post("/api/v1/finetuning/raft/export")
    def export_raft_dataset(request: RaftExperimentRequest) -> Response:
        chunks = state.store.all_chunks()
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Index documents before exporting RAFT data",
            )

        examples = make_raft_examples(
            chunks,
            questions_per_chunk=1,
            distractor_count=request.distractor_count,
        )[: request.max_examples]
        records = make_lora_records(examples)
        jsonl = make_lora_jsonl(records)
        state.logger.event(
            "finetuning.raft.exported",
            raft_examples=len(examples),
            lora_records=len(records),
        )
        return Response(
            content=jsonl,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": 'attachment; filename="finllm-raft-lora.jsonl"'
            },
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


def _sec_metadata_payload(metadata: SecFilingMetadata) -> dict[str, str]:
    return {
        "source_url": metadata.source_url,
        "ticker": metadata.ticker,
        "company": metadata.company,
        "form_type": metadata.form_type,
        "filing_date": metadata.filing_date,
    }


def _indexed_documents(chunks: list[DocumentChunk]) -> list[dict[str, object]]:
    documents: dict[DocumentInventoryKey, dict[str, object]] = {}
    for chunk in chunks:
        metadata = chunk.metadata
        key = (
            metadata.ticker,
            metadata.company,
            metadata.form_type,
            metadata.filing_date,
            metadata.source_url or metadata.source,
        )
        if key not in documents:
            documents[key] = {
                "ticker": metadata.ticker,
                "company": metadata.company,
                "form_type": metadata.form_type,
                "filing_date": metadata.filing_date,
                "source": metadata.source_url or metadata.source,
                "chunk_count": 0,
            }
        chunk_count = documents[key]["chunk_count"]
        documents[key]["chunk_count"] = (chunk_count if isinstance(chunk_count, int) else 0) + 1
    return sorted(
        documents.values(),
        key=lambda row: (str(row["ticker"]), str(row["filing_date"]), str(row["source"])),
    )


def _answer_for_eval(
    agent: ResearchAgent, question: str, filters: dict[str, str] | None
) -> tuple[str, list[str], list[str]]:
    response = agent.answer(question, filters=filters)
    return (
        response.answer,
        [result.chunk.chunk_id for result in response.retrieved_chunks],
        [citation.marker for citation in response.citations],
    )


_INPUT_TOKEN_USD = 3e-6
_OUTPUT_TOKEN_USD = 1.5e-5
_CHARS_PER_TOKEN = 4


def _evaluate_modes(state: AppState) -> list[dict[str, object]]:
    mode_results: list[dict[str, object]] = []
    for mode in ["basic_rag", "rag_rerank", "self_verify"]:
        latencies: list[float] = []
        retrieval_scores: list[float] = []
        recall_scores: list[float] = []
        mrr_scores: list[float] = []
        context_scores: list[float] = []
        citation_scores: list[float] = []
        hallucination_scores: list[float] = []
        relevance_scores: list[float] = []
        tool_successes: list[bool] = []
        costs: list[float] = []

        agent = ResearchAgent(state.store, mode=mode)
        for case in SAMPLE_EVAL_CASES:
            started = perf_counter()
            response = agent.answer(case.question, filters=case.filters)
            latencies.append(round((perf_counter() - started) * 1_000, 4))
            retrieved_ids = [result.chunk.chunk_id for result in response.retrieved_chunks]
            evidence = [result.chunk.text for result in response.retrieved_chunks]
            allowed_markers = [citation.marker for citation in response.citations]

            retrieval_scores.append(retrieval_precision(retrieved_ids, case.expected_chunk_ids))
            recall_scores.append(retrieval_recall_at_k(retrieved_ids, case.expected_chunk_ids, k=5))
            mrr_scores.append(mean_reciprocal_rank(retrieved_ids, case.expected_chunk_ids))
            context_scores.append(context_relevance(case.question, evidence))
            citation_scores.append(citation_correctness(response.answer, allowed_markers))
            hallucination_scores.append(hallucination_rate(response.answer, evidence))
            relevance_scores.append(answer_relevance(case.question, response.answer))
            tool_successes.extend(call.ok for call in response.tool_calls)
            costs.append(_estimate_query_cost(case.question, evidence, response.answer))

        mode_results.append(
            {
                "mode": mode,
                "retrieval_precision": _average(retrieval_scores),
                "retrieval_recall_at_5": _average(recall_scores),
                "retrieval_mrr": _average(mrr_scores),
                "context_relevance": _average(context_scores),
                "citation_correctness": _average(citation_scores),
                "hallucination_rate": _average(hallucination_scores),
                "answer_relevance": _average(relevance_scores),
                "avg_latency_ms": _average(latencies),
                "estimated_cost_usd": round(_average(costs), 6),
                "tool_call_success_rate": tool_call_success_rate(tool_successes),
            }
        )
        mode_results[-1]["quality_score"] = _quality_score(mode_results[-1])
    return mode_results


def _quality_score(result: dict[str, object]) -> float:
    citation = _float_metric(result["citation_correctness"])
    recall = _float_metric(result["retrieval_recall_at_5"])
    relevance = _float_metric(result["answer_relevance"])
    faithfulness = 1 - _float_metric(result["hallucination_rate"])
    score = (0.35 * citation) + (0.25 * recall) + (0.2 * relevance) + (0.2 * faithfulness)
    return round(max(0.0, min(score, 1.0)), 4)


def _best_mode(mode_results: list[dict[str, object]]) -> str:
    if not mode_results:
        return ""
    best = max(
        mode_results,
        key=lambda result: (
            _float_metric(result.get("quality_score", 0.0)),
            -_float_metric(result.get("avg_latency_ms", 0.0)),
        ),
    )
    return str(best["mode"])


def _float_metric(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return 0.0


def _estimate_query_cost(question: str, evidence: list[str], answer: str) -> float:
    input_chars = len(question) + sum(len(text) for text in evidence)
    output_chars = len(answer)
    input_tokens = input_chars / _CHARS_PER_TOKEN
    output_tokens = output_chars / _CHARS_PER_TOKEN
    return input_tokens * _INPUT_TOKEN_USD + output_tokens * _OUTPUT_TOKEN_USD


def _missing_eval_chunks(state: AppState) -> list[str]:
    indexed_ids = {chunk.chunk_id for chunk in state.store.all_chunks()}
    expected_ids = {
        chunk_id
        for case in SAMPLE_EVAL_CASES
        for chunk_id in case.expected_chunk_ids
    }
    return sorted(expected_ids - indexed_ids)


def _infer_single_company_filter(question: str, state: AppState) -> dict[str, str] | None:
    normalized_question = question.lower()
    matches: set[str] = set()
    for chunk in state.store.all_chunks():
        ticker = (chunk.metadata.ticker or "").upper()
        company = (chunk.metadata.company or "").lower()
        if not ticker:
            continue
        if _mentions_ticker(normalized_question, ticker) or (
            company and company in normalized_question
        ):
            matches.add(ticker)
    if len(matches) != 1:
        return None
    return {"ticker": next(iter(matches))}


def _mentions_ticker(normalized_question: str, ticker: str) -> bool:
    return f" {ticker.lower()} " in f" {normalized_question} "


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
