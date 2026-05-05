from __future__ import annotations


def simulate_finetuned_eval(*, model_id: str, eval_cases: int) -> dict[str, object]:
    return {
        "model_id": model_id,
        "eval_cases": eval_cases,
        "faithfulness": None,
        "citation_correctness": None,
        "limitations": ["Metrics are unset because no fine-tuned model was trained."],
    }
