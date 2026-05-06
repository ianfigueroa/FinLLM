# FinLLM Research Agent

FinLLM Research Agent is a local financial research workbench. It ingests filings and other financial documents, retrieves evidence, answers with citations, runs a few analysis tools, and scores whether answers are grounded in the retrieved text.

This is not meant to be a shiny chatbot. The useful part is the pipeline around the model: ingestion, chunk metadata, retrieval, citation checks, tool traces, latency/cost tracking, and regression evals.

## Current Status

The project is a working local MVP. It can:

- ingest the bundled sample filing;
- ingest SEC archive URLs, including `sec.gov/ix?doc=...` links;
- auto-detect SEC ticker, company, form type, and filing period from Inline XBRL metadata;
- upload `.txt` files and text-based `.pdf` filings;
- retrieve chunks with metadata;
- answer with citations that can be inspected in the UI;
- compare basic RAG, RAG with reranking, and self-verification mode;
- run calculator, metadata SQL, local market data, simple backtest, ratio, and restricted Python-analysis tools;
- build a source-backed 3-statement financial model scaffold from indexed filing chunks;
- export the model table to CSV from the UI;
- generate RAFT-style examples, export LoRA-style JSONL records, and simulate LoRA reports;
- configure optional OpenAI-compatible embeddings, Ollama generation, or OpenAI-compatible chat generation;
- persist eval run history for quality and best-mode tracking;
- run local evals for retrieval precision, citation correctness, hallucination proxy, relevance, latency, cost, and tool success.

The deliberately honest limits:

- the answer writer is conservative and extractive, not a hosted production LLM;
- Self-RAG is implemented as a local verification/critique pass, not a trained Self-RAG model;
- LoRA training is simulated unless you wire in real compute;
- scanned/image-only PDFs need OCR first;
- the eval set is small and local, not an expert-labeled benchmark.

## Running Locally

Backend:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
py -3 -m uvicorn api.main:app --host 127.0.0.1 --port 8010
```

Frontend:

```powershell
cd frontend
npm install
$env:VITE_API_BASE="http://127.0.0.1:8010"
npm run dev -- --host 127.0.0.1 --port 5180 --strictPort
```

Open `http://127.0.0.1:5180`.

Port `8000` is common enough to collide with other local projects, so `8010` is the safer default for manual runs.

## Running With Docker

Docker is useful here because it gives you a repeatable API/UI setup and persistent local Chroma storage without relying on whatever Python or Node packages are already on your machine.

```powershell
docker compose -f infra/docker-compose.yml up --build
```

That starts:

- API: `http://127.0.0.1:8000`
- UI: `http://127.0.0.1:5173`
- Chroma-backed storage in the `finllm-storage` Docker volume

The containers include health checks, and the frontend waits for the API to become healthy.

## What The UI Buttons Do

`Load demo sample` loads `examples/sample_docs/acme_10k_2025.txt`, chunks it, embeds it, and stores it. The built-in evals use this sample, so run this before `Run eval`.

`Index SEC filing` fetches a filing from SEC archives, detects filing metadata from Inline XBRL, cleans the HTML, chunks it, and stores it with ticker/company/form/date metadata. This works with Inline XBRL URLs such as:

```text
https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm
```

`Upload text or PDF filing` indexes a local `.txt` or text-based `.pdf`. PDFs with only scanned images are rejected because there is no OCR layer yet.

`Run eval` runs the local regression/eval suite. It compares the implemented modes and reports retrieval, citation, relevance, latency, cost, tool-call, quality-score, and best-mode metrics.

`Generate RAFT data` creates training-style examples from indexed chunks. Each example includes a question, relevant evidence, distractors, a cited answer, and metadata. The LoRA step is simulated so the data shape can be checked without a GPU.

`Export JSONL` downloads LoRA-style instruction records as `finllm-raft-lora.jsonl`. That file is the handoff point for real fine-tuning experiments.

`Build model` extracts supported income statement, balance sheet, and cash flow lines
from the active ticker's indexed chunks, attaches model-source citations, projects the
next one to five years from configurable revenue growth, and renders the result in a
downloadable CSV table. It is a modeling scaffold, not an audited XBRL model.

## API Endpoints

- `POST /api/v1/ingestions/sample`
- `POST /api/v1/ingestions/sec-url`
- `POST /api/v1/ingestions/sec-url/metadata`
- `POST /api/v1/documents/upload`
- `GET /api/v1/ingestions/status`
- `POST /api/v1/chat`
- `POST /api/v1/evals`
- `GET /api/v1/evals/history`
- `POST /api/v1/finetuning/raft`
- `POST /api/v1/finetuning/raft/export`
- `POST /api/v1/models/three-statement`
- `GET /api/v1/system/status`

Example chat request:

```json
{
  "question": "What risk factors did Acme disclose?",
  "mode": "rag_rerank"
}
```

Example SEC ingestion request:

```json
{
  "url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581026000021/nvda-20260125.htm",
  "ticker": "NVDA",
  "company": "NVIDIA",
  "form_type": "10-K",
  "filing_date": "2026-01-25"
}
```

## Architecture

```text
ingestion/     loaders, cleaning, chunking, upload parsing, metadata
retrieval/     embeddings, Chroma/in-memory stores, hybrid search, reranking, citations
agent/         planner, workflow, tools, memory, verifier
evals/         retrieval/citation/relevance/hallucination/cost/latency checks
finetuning/    RAFT data, LoRA-format data, simulated training/eval reports
api/           FastAPI app and schemas
frontend/      React + TypeScript workbench
modeling/      3-statement extraction, source mapping, and projection scaffold
infra/         Docker Compose, Dockerfiles, AWS notes
reports/       research report and eval summaries
tests/         unit and integration tests
```

## Research Thread

The main question is:

> How do vanilla RAG, Self-RAG-style critique, RAFT-style training data, and optional LoRA fine-tuning compare for financial research tasks in citation faithfulness, answer accuracy, latency, and cost?

The implementation supports that comparison in a local, measurable way:

- `basic_rag`: direct vector retrieval.
- `rag_rerank`: hybrid retrieval plus reranking.
- `self_verify`: reranked retrieval plus citation verification and limitations.
- `RAFT`: dataset generation, JSONL export, plus simulated LoRA report.
- `Modeling`: source-backed 3-statement scaffold with cited historical lines and
  configurable projection assumptions.

Cost is tracked because it matters once hosted LLMs or paid rerankers are plugged in. In the current local setup, estimated cost is `$0.00`.

## Verification

Latest local verification:

- `ruff`: passed
- `mypy`: passed
- `pytest`: `108 passed`
- coverage: `93.78%`
- frontend build: passed
- `pip-audit`: no known vulnerabilities

`npm audit` needs a committed frontend lockfile. The project currently avoids that large generated file to keep history small.

## Reports

- Research report: `reports/research_report.md`
- Eval summary: `reports/eval_results.md`

The app is research software, not financial advice. Outputs should be treated as cited analysis over the indexed corpus.
