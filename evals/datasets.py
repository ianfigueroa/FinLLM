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
        case_id="acme-risk-factors",
        question="What customer concentration risk did Acme disclose?",
        expected_chunk_ids=["ACME-10-K-2025-02-15-0001"],
        expected_terms=["customer", "risk"],
        filters={"ticker": "ACME"},
    )
]
