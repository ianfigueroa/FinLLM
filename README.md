# FinLLM Research Agent

FinLLM Research Agent is a retrieval-augmented financial research platform for cited answers, tool-backed analysis, and hallucination/citation evaluation. The first implementation is offline-capable by default: deterministic embeddings and an in-memory vector store make tests and demos reproducible without paid APIs.

## What It Builds Toward

Research question:

> How do vanilla RAG, Self-RAG-style critique, RAFT-style training data, and optional LoRA fine-tuning compare for financial research tasks in citation faithfulness, answer accuracy, latency, and cost?

Core modes:

1. Basic RAG.
2. RAG with reranking.
3. RAG with self-verification and critique.
4. RAFT-style dataset generation with optional fine-tuning hooks.

## Architecture

```text
ingestion/     document loading, cleaning, chunking, metadata
retrieval/     embeddings, vector storage, hybrid search, reranking, citations
agent/         controlled workflow, tools, prompts, verifier, memory
evals/         retrieval, citation, faithfulness, cost, latency regression checks
finetuning/    RAFT and LoRA dataset generation plus simulated training reports
api/           FastAPI app with chat, upload, eval, and ingestion status endpoints
frontend/      React + TypeScript research UI with citations and dashboards
infra/         Docker Compose and deployment notes
reports/       research report and eval result summaries
tests/         unit and integration coverage
```

## Local Setup

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m pytest
uvicorn api.main:app --reload
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

The API defaults to deterministic local embeddings and an in-memory vector store for reproducible tests. Use `retrieval/chroma_store.py` when persistent local vector storage is needed.

## API Workflow

Start the backend:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

If another local project already owns port `8000`, use a different API port and point Vite at it:

```powershell
py -3 -m uvicorn api.main:app --host 127.0.0.1 --port 8010
cd frontend
$env:VITE_API_BASE="http://127.0.0.1:8010"
npm run dev -- --host 127.0.0.1 --port 5180 --strictPort
```

Useful endpoints:

- `POST /api/v1/ingestions/sample` indexes the bundled Acme 10-K sample.
- `POST /api/v1/ingestions/sec-url` fetches and indexes an SEC archive filing URL.
- `POST /api/v1/documents/upload` indexes a plain-text filing or transcript with metadata.
- `POST /api/v1/chat` answers with citations and retrieved chunks.
- `POST /api/v1/evals` runs the local regression harness.
- `GET /api/v1/ingestions/status` returns indexed chunk count.

Example chat payload:

```json
{
  "question": "What risk factors did Acme disclose?",
  "mode": "rag_rerank",
  "filters": { "ticker": "ACME" }
}
```

Example SEC URL ingestion payload:

```json
{
  "url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm",
  "ticker": "NVDA",
  "company": "NVIDIA",
  "form_type": "10-K",
  "filing_date": "2026-01-25"
}
```

## Verification Snapshot

Current local checks:

- Python tests: 56 passed.
- Python coverage: 95.53%.
- Ruff and MyPy: passed.
- Chroma vector store integration: passed.
- Frontend build: passed.
- Security audit note: run dependency audits in a clean project environment with a lockfile before deployment.

See `reports/eval_results.md` for the evaluation summary.

## Design Principles

- Answers must cite retrieved chunks. Unsupported answers should say evidence is insufficient.
- Facts, inference, confidence, limitations, and "not financial advice" framing are separated.
- Retrieval, prompts, tool calls, latency, estimated cost, and verification quality are structured logs.
- APIs validate user input and do not log secrets or uploaded document contents.
- Evaluation is a first-class workflow, not an afterthought.

## Paper Inspiration

- RAG: retrieve external evidence before generation.
- Self-RAG: critique and verify whether the answer is supported by context.
- RAFT: train on relevant evidence and distractor chunks so the model learns grounded answering.
- RAGAS-style metrics: faithfulness, answer relevance, and context relevance.

The detailed write-up lives in `reports/research_report.md`.

## Docker Compose

```powershell
docker compose -f infra/docker-compose.yml up --build
```

This runs the FastAPI backend on `http://127.0.0.1:8000` and the Vite frontend on `http://127.0.0.1:5173`.
