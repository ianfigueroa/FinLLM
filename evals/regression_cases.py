from __future__ import annotations

from collections.abc import Callable

from evals.citation_eval import citation_correctness
from evals.datasets import EvalCase
from evals.ragas_eval import retrieval_precision

AnswerFn = Callable[[EvalCase], tuple[str, list[str], list[str]]]


def run_regression_cases(cases: list[EvalCase], answer_fn: AnswerFn) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in cases:
        answer, retrieved_ids, allowed_markers = answer_fn(case)
        retrieval_score = retrieval_precision(retrieved_ids, case.expected_chunk_ids)
        citation_score = citation_correctness(answer, allowed_markers)
        term_passed = all(term.lower() in answer.lower() for term in case.expected_terms)
        results.append(
            {
                "case_id": case.case_id,
                "retrieval_precision": retrieval_score,
                "citation_correctness": citation_score,
                "passed": retrieval_score == 1.0 and citation_score == 1.0 and term_passed,
            }
        )
    return results
