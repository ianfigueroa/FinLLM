"""Multi-filing retrieval/answer evaluation over real SEC 10-Ks.

Builds a corpus from the bundled NVDA sample plus three live 10-Ks (Apple, Microsoft,
Amazon) pulled through the same SEC loader the app uses, then scores all three pipeline
modes on a labeled question set:

  * financial facts  - the gold chunk is located by the exact reported figure, which
    never appears in the question, so retrieval is genuine (not lexical give-away);
  * qualitative      - NVDA risk-factor / segment / driver questions, hand-labeled.

Metrics reuse the repo's own eval functions. Requires network access to www.sec.gov;
if a filing cannot be fetched it is skipped and the run continues on what loaded.

    PYTHONPATH=. python evals/real_corpus_eval.py
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean
from time import perf_counter

from agent.graph import ResearchAgent
from evals.citation_eval import citation_correctness
from evals.datasets import EvalCase
from evals.hallucination_eval import hallucination_rate
from evals.ragas_eval import answer_relevance, retrieval_precision
from evals.retrieval_eval import mean_reciprocal_rank, retrieval_recall_at_k
from ingestion.chunker import chunk_document
from ingestion.sec_loader import load_sec_filing
from ingestion.sec_url_loader import load_sec_filing_url
from retrieval.embeddings import build_embedding_model, load_embedding_settings
from retrieval.vector_store import InMemoryVectorStore

LIVE_FILINGS = [
    ("AAPL", "Apple Inc.", "2025-09-27",
     "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"),
    ("MSFT", "Microsoft Corporation", "2026-07-29",
     "https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm"),
    ("AMZN", "Amazon.com Inc.", "2026-02-06",
     "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm"),
]

# (ticker, metric phrasing, exact current-year figure used only to locate the gold chunk)
FIN_FACTS = [
    ("AAPL", "total revenue", "416161000000"),
    ("AAPL", "gross profit", "195201000000"),
    ("AAPL", "operating income", "133050000000"),
    ("AAPL", "net income", "112010000000"),
    ("AAPL", "total assets", "359241000000"),
    ("AAPL", "cash from operating activities", "111482000000"),
    ("AAPL", "capital expenditures", "12715000000"),
    ("MSFT", "total revenue", "331839000000"),
    ("MSFT", "gross profit", "225465000000"),
    ("MSFT", "operating income", "155237000000"),
    ("MSFT", "net income", "133749000000"),
    ("MSFT", "total assets", "758376000000"),
    ("MSFT", "capital expenditures", "115948000000"),
    ("AMZN", "total revenue", "716924000000"),
    ("AMZN", "operating income", "79975000000"),
    ("AMZN", "net income", "77670000000"),
    ("AMZN", "total assets", "818042000000"),
    ("AMZN", "cash from operating activities", "139514000000"),
    ("AMZN", "capital expenditures", "131819000000"),
]

NCID = "NVDA-10-K-2026-02-21-{:04d}".format
NVDA_CASES = [
    ("cust-concentration", "What customer concentration risk did NVIDIA disclose?", NCID(5), ["customer", "data center"]),
    ("suppliers", "Which foundry and packaging suppliers does NVIDIA rely on?", NCID(6), ["tsmc", "packaging"]),
    ("export-controls", "How do U.S. export controls to China affect NVIDIA?", NCID(7), ["export controls", "china"]),
    ("segments", "What are NVIDIA's two reportable segments?", NCID(2), ["compute", "graphics"]),
    ("net-revenue", "What was NVIDIA's net revenue for fiscal 2026?", NCID(14), ["130,497"]),
    ("gross-profit", "What was NVIDIA gross profit for fiscal 2026?", NCID(15), ["97,873"]),
    ("net-income", "What was NVIDIA net income for fiscal 2026?", NCID(15), ["72,880"]),
    ("operating-income", "What was NVIDIA operating income for fiscal 2026?", NCID(15), ["81,453"]),
    ("total-liabilities", "What were NVIDIA total liabilities at year end?", NCID(16), ["30,109"]),
    ("op-cash-flow", "What was NVIDIA net cash provided by operating activities?", NCID(16), ["64,089"]),
    ("capex", "What were NVIDIA capital expenditures?", NCID(16), ["3,236"]),
    ("architectures", "Which GPU architectures outpaced supply in fiscal 2026?", NCID(3), ["hopper", "blackwell"]),
    ("gross-margin", "Why did NVIDIA gross margin expand year over year?", NCID(11), ["gross margin", "blackwell"]),
    ("cuda-lockin", "What risk relates to customers reducing CUDA lock-in?", NCID(8), ["cuda"]),
    ("networking", "Which networking products did NVIDIA expand with Blackwell?", NCID(4), ["spectrum-x"]),
]

COMPANY_FIRST = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon"}


def build_store() -> tuple[InMemoryVectorStore, set[str]]:
    store = InMemoryVectorStore(build_embedding_model(load_embedding_settings()))
    loaded = {"NVDA"}
    nvda = load_sec_filing(
        Path("examples/sample_docs/nvda_10k_2026.txt"),
        ticker="NVDA", company="NVIDIA Corporation", form_type="10-K", filing_date="2026-02-21",
    )
    store.upsert(chunk_document(nvda, max_chars=420, overlap_chars=60))
    for ticker, company, date, url in LIVE_FILINGS:
        try:
            doc = load_sec_filing_url(url, ticker=ticker, company=company, form_type="10-K", filing_date=date)
            store.upsert(chunk_document(doc, max_chars=420, overlap_chars=60))
            loaded.add(ticker)
        except Exception as exc:  # network / SEC availability
            print(f"  (skipped {ticker}: {type(exc).__name__})")
    return store, loaded


def build_cases(store: InMemoryVectorStore, loaded: set[str]) -> list[EvalCase]:
    by_ticker: dict[str, list] = {}
    for chunk in store.all_chunks():
        by_ticker.setdefault(chunk.metadata.ticker or "", []).append(chunk)

    cases: list[EvalCase] = []
    for ticker, metric, figure in FIN_FACTS:
        if ticker not in loaded:
            continue
        gold = [c.chunk_id for c in by_ticker[ticker] if figure in c.text]
        if not gold:
            continue
        cases.append(EvalCase(
            case_id=f"{ticker.lower()}-{metric.replace(' ', '-')}",
            question=f"What was {COMPANY_FIRST[ticker]}'s {metric} in the latest fiscal year?",
            expected_chunk_ids=gold, expected_terms=[], filters={"ticker": ticker},
        ))
    for cid, question, gold, terms in NVDA_CASES:
        cases.append(EvalCase(cid, question, [gold], terms, {"ticker": "NVDA"}))
    return cases


def hit_at_k(retrieved: list[str], expected: list[str], k: int = 5) -> float:
    return 1.0 if set(retrieved[:k]) & set(expected) else 0.0


def evaluate(store: InMemoryVectorStore, cases: list[EvalCase], mode: str) -> dict[str, float]:
    agent = ResearchAgent(store, mode=mode)
    hit, prec, rec, mrr, cite, halluc, rel, lat = ([] for _ in range(8))
    for case in cases:
        started = perf_counter()
        resp = agent.answer(case.question, filters=case.filters)
        lat.append((perf_counter() - started) * 1000)
        ids = [r.chunk.chunk_id for r in resp.retrieved_chunks]
        evidence = [r.chunk.text for r in resp.retrieved_chunks]
        markers = [c.marker for c in resp.citations]
        hit.append(hit_at_k(ids, case.expected_chunk_ids))
        prec.append(retrieval_precision(ids, case.expected_chunk_ids))
        rec.append(retrieval_recall_at_k(ids, case.expected_chunk_ids, k=5))
        mrr.append(mean_reciprocal_rank(ids, case.expected_chunk_ids))
        cite.append(citation_correctness(resp.answer, markers))
        halluc.append(hallucination_rate(resp.answer, evidence))
        rel.append(answer_relevance(case.question, resp.answer))
    return {
        "hit@5": mean(hit), "precision": mean(prec), "recall@5": mean(rec), "mrr": mean(mrr),
        "citation": mean(cite), "halluc": mean(halluc), "answer_rel": mean(rel), "lat_ms": mean(lat),
    }


def main() -> None:
    store, loaded = build_store()
    cases = build_cases(store, loaded)
    print(f"\ncorpus: {len(loaded)} filings {sorted(loaded)}, {len(store.all_chunks())} chunks, "
          f"{len(cases)} labeled questions\n")
    cols = ["mode", "hit@5", "precision", "recall@5", "mrr", "citation", "halluc", "answer_rel", "lat_ms"]
    print("  ".join(f"{c:>10}" for c in cols))
    for mode in ("basic_rag", "rag_rerank", "self_verify"):
        m = evaluate(store, cases, mode)
        row = [mode, m["hit@5"], m["precision"], m["recall@5"], m["mrr"],
               m["citation"], m["halluc"], m["answer_rel"], m["lat_ms"]]
        print("  ".join(f"{row[0]:>10}" if i == 0 else f"{v:>10.3f}" for i, v in enumerate(row)))


if __name__ == "__main__":
    main()
