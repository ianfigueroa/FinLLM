from __future__ import annotations

from retrieval.citations import extract_markers


def citation_correctness(answer: str, allowed_markers: list[str]) -> float:
    used_markers = extract_markers(answer)
    if not used_markers:
        return 0.0
    allowed = set(allowed_markers)
    correct = sum(1 for marker in used_markers if marker in allowed)
    return round(correct / len(used_markers), 4)

