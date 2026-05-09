from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    expected_chunk_ids: list[str]
    expected_terms: list[str]
    filters: dict[str, str] = field(default_factory=dict)


SAMPLE_EVAL_CASES = [
    EvalCase(
        case_id="nvda-customer-concentration",
        question="What customer concentration risk did NVIDIA disclose in its risk factors?",
        expected_chunk_ids=["NVDA-10-K-2026-02-21-0005"],
        expected_terms=["customer", "data center"],
        filters={"ticker": "NVDA"},
    )
]
