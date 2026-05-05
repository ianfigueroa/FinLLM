from __future__ import annotations


def tool_call_success_rate(successes: list[bool]) -> float:
    if not successes:
        return 0.0
    return round(sum(1 for success in successes if success) / len(successes), 4)
