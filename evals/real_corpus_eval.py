"""Multi-filing retrieval/answer evaluation over real SEC 10-Ks.

Builds a corpus from the bundled NVDA sample plus three real 10-Ks (Apple, Microsoft,
Amazon) and scores all three pipeline modes on a labeled question set:

  * financial facts  - the gold chunks are located by the exact reported figure, which
    never appears in the question, so retrieval is genuine (not lexical give-away);
  * qualitative      - NVDA risk-factor / segment / driver questions, hand-labeled.

Filings are read from FINLLM_CORPUS_CACHE (default stuff/corpus_cache/<TICKER>.json) when
present. Otherwise they are fetched from www.sec.gov through the app's SEC loader (set
FINLLM_SEC_USER_AGENT to identify yourself) and written to the cache. A filing that cannot
be loaded is skipped and the run continues on what loaded.

Questions are split into DEV (the original 34 plus half of the added ones) and TEST (the
other half). Tune on DEV only; TEST is for the final report.

    PYTHONPATH=. python evals/real_corpus_eval.py [--splits dev,test,original] [--no-prefixes]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

from agent.graph import ResearchAgent
from evals.citation_eval import citation_correctness
from evals.datasets import EvalCase
from evals.hallucination_eval import hallucination_rate
from evals.ragas_eval import answer_relevance, retrieval_precision
from evals.retrieval_eval import mean_reciprocal_rank, retrieval_recall_at_k
from ingestion.chunker import _section_for_chunk, chunk_document
from ingestion.metadata import DocumentChunk, DocumentMetadata
from ingestion.sec_loader import load_sec_filing
from ingestion.sec_url_loader import load_sec_filing_url
from retrieval.embeddings import build_embedding_model, load_embedding_settings
from retrieval.vector_store import InMemoryVectorStore

CACHE_DIR = Path(os.getenv("FINLLM_CORPUS_CACHE", "stuff/corpus_cache"))
MODES = ("basic_rag", "rag_rerank", "self_verify")

LIVE_FILINGS = [
    ("AAPL", "Apple Inc.", "2025-09-27",
     "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"),
    ("MSFT", "Microsoft Corporation", "2026-07-29",
     "https://www.sec.gov/Archives/edgar/data/789019/000119312526323660/msft-20260630.htm"),
    ("AMZN", "Amazon.com Inc.", "2026-02-06",
     "https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm"),
]

# (ticker, metric phrasing, exact current-year XBRL figure used only to locate the gold chunk)
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

# (ticker, metric phrasing, current-year figure in millions as printed in the filing's
# statements and notes). Gold chunks are every chunk that prints the figure, in either
# the "34,550" form or the raw XBRL form "34550000000".
MORE_FIN_FACTS = [
    ("AAPL", "products net sales", "307,003"),
    ("AAPL", "services net sales", "109,158"),
    ("AAPL", "iPhone net sales", "209,586"),
    ("AAPL", "Mac net sales", "33,708"),
    ("AAPL", "iPad net sales", "28,023"),
    ("AAPL", "wearables, home and accessories net sales", "35,686"),
    ("AAPL", "Europe segment net sales", "111,032"),
    ("AAPL", "Greater China segment net sales", "64,377"),
    ("AAPL", "total cost of sales", "220,960"),
    ("AAPL", "research and development expense", "34,550"),
    ("AAPL", "selling, general and administrative expense", "27,601"),
    ("AAPL", "total operating expenses", "62,151"),
    ("AAPL", "income before provision for income taxes", "132,729"),
    ("AAPL", "provision for income taxes", "20,719"),
    ("AAPL", "accounts receivable", "39,777"),
    ("AAPL", "inventories", "5,718"),
    ("AAPL", "total current assets", "147,957"),
    ("AAPL", "property, plant and equipment, net", "49,834"),
    ("AAPL", "accounts payable", "69,860"),
    ("AAPL", "total current liabilities", "165,631"),
    ("AAPL", "depreciation and amortization", "11,698"),
    ("AAPL", "share-based compensation expense", "12,863"),
    ("AAPL", "dividends paid", "15,421"),
    ("AAPL", "repurchases of common stock", "90,711"),
    ("MSFT", "product revenue", "64,696"),
    ("MSFT", "service and other revenue", "267,143"),
    ("MSFT", "total cost of revenue", "106,374"),
    ("MSFT", "research and development expense", "35,562"),
    ("MSFT", "sales and marketing expense", "26,710"),
    ("MSFT", "general and administrative expense", "7,956"),
    ("MSFT", "Intelligent Cloud revenue", "137,791"),
    ("MSFT", "income before income taxes", "165,934"),
    ("MSFT", "provision for income taxes", "32,185"),
    ("MSFT", "accounts receivable", "80,876"),
    ("MSFT", "total current assets", "207,710"),
    ("MSFT", "property and equipment, net", "313,076"),
    ("MSFT", "goodwill", "119,651"),
    ("MSFT", "accounts payable", "42,416"),
    ("MSFT", "short-term unearned revenue", "72,965"),
    ("MSFT", "total current liabilities", "168,825"),
    ("MSFT", "long-term debt", "31,067"),
    ("MSFT", "total liabilities", "315,989"),
    ("MSFT", "retained earnings", "328,265"),
    ("MSFT", "depreciation, amortization, and other", "38,534"),
    ("MSFT", "stock-based compensation expense", "12,405"),
    ("MSFT", "net cash from operations", "182,935"),
    ("MSFT", "common stock repurchased", "22,271"),
    ("MSFT", "dividends paid", "26,445"),
    ("AMZN", "net product sales", "296,266"),
    ("AMZN", "net service sales", "420,658"),
    ("AMZN", "cost of sales", "356,414"),
    ("AMZN", "fulfillment expense", "109,074"),
    ("AMZN", "technology and infrastructure expense", "108,521"),
    ("AMZN", "sales and marketing expense", "47,129"),
    ("AMZN", "total operating expenses", "636,949"),
    ("AMZN", "AWS segment net sales", "128,725"),
    ("AMZN", "AWS segment operating income", "45,606"),
    ("AMZN", "North America segment net sales", "426,305"),
    ("AMZN", "International segment net sales", "161,894"),
    ("AMZN", "online stores net sales", "269,287"),
    ("AMZN", "third-party seller services net sales", "172,162"),
    ("AMZN", "advertising services net sales", "68,635"),
    ("AMZN", "subscription services net sales", "49,619"),
    ("AMZN", "income before income taxes", "97,311"),
    ("AMZN", "provision for income taxes", "19,087"),
    ("AMZN", "depreciation and amortization", "65,756"),
    ("AMZN", "stock-based compensation expense", "19,467"),
    ("AMZN", "inventories", "38,325"),
    ("AMZN", "property and equipment, net", "357,025"),
    ("AMZN", "accounts payable", "121,909"),
    ("AMZN", "total current liabilities", "218,005"),
    ("AMZN", "long-term debt", "65,648"),
    ("AMZN", "retained earnings", "250,536"),
    ("AMZN", "free cash flow", "11,194"),
]

NCID = "NVDA-10-K-2026-02-21-{:04d}".format
NVDA_CASES = [
    ("cust-concentration", "What customer concentration risk did NVIDIA disclose?", NCID(5),
     ["customer", "data center"]),
    ("suppliers", "Which foundry and packaging suppliers does NVIDIA rely on?", NCID(6),
     ["tsmc", "packaging"]),
    ("export-controls", "How do U.S. export controls to China affect NVIDIA?", NCID(7),
     ["export controls", "china"]),
    ("segments", "What are NVIDIA's two reportable segments?", NCID(2), ["compute", "graphics"]),
    ("net-revenue", "What was NVIDIA's net revenue for fiscal 2026?", NCID(14), ["130,497"]),
    ("gross-profit", "What was NVIDIA gross profit for fiscal 2026?", NCID(15), ["97,873"]),
    ("net-income", "What was NVIDIA net income for fiscal 2026?", NCID(15), ["72,880"]),
    ("operating-income", "What was NVIDIA operating income for fiscal 2026?", NCID(15),
     ["81,453"]),
    ("total-liabilities", "What were NVIDIA total liabilities at year end?", NCID(16),
     ["30,109"]),
    ("op-cash-flow", "What was NVIDIA net cash provided by operating activities?", NCID(16),
     ["64,089"]),
    ("capex", "What were NVIDIA capital expenditures?", NCID(16), ["3,236"]),
    ("architectures", "Which GPU architectures outpaced supply in fiscal 2026?", NCID(3),
     ["hopper", "blackwell"]),
    ("gross-margin", "Why did NVIDIA gross margin expand year over year?", NCID(11),
     ["gross margin", "blackwell"]),
    ("cuda-lockin", "What risk relates to customers reducing CUDA lock-in?", NCID(8), ["cuda"]),
    ("networking", "Which networking products did NVIDIA expand with Blackwell?", NCID(4),
     ["spectrum-x"]),
]

# Each answer appears in exactly one chunk of the bundled sample.
MORE_NVDA_CASES = [
    ("revenue-drivers", "What drove NVIDIA's revenue growth in fiscal 2026?", NCID(9),
     ["data center", "hopper"]),
    ("gaming-revenue", "What helped NVIDIA's Gaming revenue?", NCID(10), ["geforce rtx"]),
    ("opex-increase", "Why did NVIDIA's operating expenses increase?", NCID(12), ["headcount"]),
    ("capital-return", "How did NVIDIA return capital to shareholders?", NCID(13),
     ["share repurchases", "dividend"]),
    ("supply-constraint", "What is the principal constraint on NVIDIA's near-term revenue?",
     NCID(14), ["supply"]),
    ("platform-users", "Who uses NVIDIA's accelerated computing platform?", NCID(1),
     ["cloud service providers", "researchers"]),
]

COMPANY_FIRST = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon"}


def _fin_case_id(ticker: str, metric: str) -> str:
    return f"{ticker.lower()}-{re.sub(r'[^a-z0-9]+', '-', metric.lower()).strip('-')}"


ORIGINAL_CASE_IDS = frozenset(
    [_fin_case_id(ticker, metric) for ticker, metric, _ in FIN_FACTS]
    + [case_id for case_id, *_ in NVDA_CASES]
)
ADDED_CASE_IDS = [_fin_case_id(ticker, metric) for ticker, metric, _ in MORE_FIN_FACTS] + [
    case_id for case_id, *_ in MORE_NVDA_CASES
]


def _split_key(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


# Added questions ordered by a hash of their id, then alternated DEV/TEST.
TEST_CASE_IDS = frozenset(sorted(ADDED_CASE_IDS, key=_split_key)[1::2])


def split_of(case_id: str) -> str:
    return "test" if case_id in TEST_CASE_IDS else "dev"


def load_filing_chunks(ticker: str, company: str, date: str, url: str) -> list[DocumentChunk]:
    cache_path = CACHE_DIR / f"{ticker}.json"
    if cache_path.exists():
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
        return _chunks_from_cache(rows, ticker=ticker, company=company, url=url)
    doc = load_sec_filing_url(url, ticker=ticker, company=company, form_type="10-K",
                              filing_date=date)
    chunks = chunk_document(doc, max_chars=420, overlap_chars=60)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([{"id": c.chunk_id, "text": c.text} for c in chunks]), encoding="utf-8"
    )
    return chunks


def _chunks_from_cache(
    rows: list[dict[str, str]], *, ticker: str, company: str, url: str
) -> list[DocumentChunk]:
    # The cache keeps chunk text only; sections are rebuilt in order the same way the
    # chunker assigns them, so cached and fetched corpora index identically.
    chunks: list[DocumentChunk] = []
    section: str | None = None
    for row in rows:
        section = _section_for_chunk(row["text"], section)
        metadata = DocumentMetadata(
            source=url, ticker=ticker, company=company, form_type="10-K",
            filing_date=row["id"][-15:-5], section=section, source_url=url,
        )
        chunks.append(DocumentChunk(row["id"], row["text"], metadata, 0, len(row["text"])))
    return chunks


def load_corpus() -> tuple[list[DocumentChunk], set[str]]:
    nvda = load_sec_filing(
        Path("examples/sample_docs/nvda_10k_2026.txt"),
        ticker="NVDA", company="NVIDIA Corporation", form_type="10-K", filing_date="2026-02-21",
    )
    chunks = chunk_document(nvda, max_chars=420, overlap_chars=60)
    loaded = {"NVDA"}
    for ticker, company, date, url in LIVE_FILINGS:
        try:
            chunks.extend(load_filing_chunks(ticker, company, date, url))
            loaded.add(ticker)
        except Exception as exc:  # network / SEC availability
            print(f"  (skipped {ticker}: {type(exc).__name__})")
    return chunks, loaded


def build_cases(chunks: list[DocumentChunk], loaded: set[str]) -> list[EvalCase]:
    by_ticker: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        by_ticker.setdefault(chunk.metadata.ticker or "", []).append(chunk)

    def question(ticker: str, metric: str) -> str:
        return f"What was {COMPANY_FIRST[ticker]}'s {metric} in the latest fiscal year?"

    cases: list[EvalCase] = []
    for ticker, metric, figure in FIN_FACTS:
        if ticker not in loaded:
            continue
        gold = [c.chunk_id for c in by_ticker[ticker] if figure in c.text]
        if gold:
            cases.append(EvalCase(_fin_case_id(ticker, metric), question(ticker, metric),
                                  gold, [], {"ticker": ticker}))
    for ticker, metric, figure in MORE_FIN_FACTS:
        if ticker not in loaded:
            continue
        raw = str(int(figure.replace(",", "")) * 1_000_000)
        pattern = re.compile(rf"(?<![\d,]){re.escape(figure)}(?!,?\d)|(?<!\d){raw}(?!\d)")
        gold = [c.chunk_id for c in by_ticker[ticker] if pattern.search(c.text)]
        if gold:
            cases.append(EvalCase(_fin_case_id(ticker, metric), question(ticker, metric),
                                  gold, [], {"ticker": ticker}))
    for case_id, text, gold_id, terms in NVDA_CASES + MORE_NVDA_CASES:
        cases.append(EvalCase(case_id, text, [gold_id], terms, {"ticker": "NVDA"}))
    return cases


def hit_at_k(retrieved: list[str], expected: list[str], k: int = 5) -> float:
    return 1.0 if set(retrieved[:k]) & set(expected) else 0.0


def evaluate(
    store: InMemoryVectorStore, cases: list[EvalCase], mode: str
) -> dict[str, dict[str, float]]:
    agent = ResearchAgent(store, mode=mode)
    rows: dict[str, dict[str, float]] = {}
    for case in cases:
        started = perf_counter()
        resp = agent.answer(case.question, filters=case.filters)
        latency = (perf_counter() - started) * 1000
        ids = [r.chunk.chunk_id for r in resp.retrieved_chunks]
        evidence = [r.chunk.text for r in resp.retrieved_chunks]
        markers = [c.marker for c in resp.citations]
        rows[case.case_id] = {
            "hit@5": hit_at_k(ids, case.expected_chunk_ids),
            "precision": retrieval_precision(ids, case.expected_chunk_ids),
            "recall@5": retrieval_recall_at_k(ids, case.expected_chunk_ids, k=5),
            "mrr": mean_reciprocal_rank(ids, case.expected_chunk_ids),
            "citation": citation_correctness(resp.answer, markers),
            "halluc": hallucination_rate(resp.answer, evidence),
            "answer_rel": answer_relevance(case.question, resp.answer),
            "lat_ms": latency,
        }
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {metric: mean(row[metric] for row in rows) for metric in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--splits", default="dev,test,original",
                        help="comma list of dev, test, original, all")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--no-prefixes", action="store_true",
                        help="disable the model's query/document prefixes (ablation)")
    args = parser.parse_args()
    splits = args.splits.split(",")

    model = build_embedding_model(load_embedding_settings())
    if args.no_prefixes and hasattr(model, "query_prefix"):
        model = dataclasses.replace(model, query_prefix="", document_prefix="")
    chunks, loaded = load_corpus()
    store = InMemoryVectorStore(model)
    started = perf_counter()
    store.upsert(chunks)
    index_s = perf_counter() - started

    cases = build_cases(chunks, loaded)
    members = {
        "dev": [c for c in cases if split_of(c.case_id) == "dev"],
        "test": [c for c in cases if split_of(c.case_id) == "test"],
        "original": [c for c in cases if c.case_id in ORIGINAL_CASE_IDS],
        "all": cases,
    }
    wanted = {c.case_id for split in splits for c in members[split]}
    prefixes = (getattr(model, "query_prefix", ""), getattr(model, "document_prefix", ""))
    print(f"\ncorpus: {len(loaded)} filings {sorted(loaded)}, {len(chunks)} chunks, "
          f"{len(cases)} labeled questions "
          f"(dev {len(members['dev'])}, test {len(members['test'])}, "
          f"original {len(members['original'])})")
    print(f"embedding: {type(model).__name__} {getattr(model, 'model', '')} "
          f"prefixes={prefixes!r} index_s={index_s:.1f}")

    cols = ["mode", "hit@5", "precision", "recall@5", "mrr", "citation", "halluc",
            "answer_rel", "lat_ms"]
    results = {
        mode: evaluate(store, [c for c in cases if c.case_id in wanted], mode)
        for mode in args.modes.split(",")
    }
    for split in splits:
        print(f"\n[{split}] n={len(members[split])}")
        print("  ".join(f"{c:>11}" for c in cols))
        for mode, rows in results.items():
            summary = summarize([rows[c.case_id] for c in members[split]])
            print(f"{mode:>11}  " + "  ".join(f"{summary[c]:>11.3f}" for c in cols[1:]))


if __name__ == "__main__":
    main()
