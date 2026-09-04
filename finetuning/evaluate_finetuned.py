from __future__ import annotations


def simulate_finetuned_eval(*, model_id: str, eval_cases: int) -> dict[str, object]:
    """Stub. There is no fine-tuned model to evaluate, so the metrics stay None."""
    return {
        "model_id": model_id,
        "eval_cases": eval_cases,
        "faithfulness": None,
        "citation_correctness": None,
        "limitations": ["Metrics are unset because no fine-tuned model was trained."],
    }
