from __future__ import annotations


def simulate_lora_training(*, records: int, base_model: str) -> dict[str, object]:
    """Stub. Nothing is trained; returns a placeholder report for the API."""
    return {
        "status": "simulated",
        "model_id": f"{base_model}-raft-lora-sim",
        "records": records,
        "limitations": [
            "No GPU training was run.",
            "Use this report to validate data shape before real LoRA training.",
        ],
    }
