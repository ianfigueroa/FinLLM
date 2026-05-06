from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    retrieval_required: bool
    tool_required: bool
    reason: str


class Planner:
    def plan(self, question: str) -> Plan:
        lowered = question.lower()
        tool_required = any(term in lowered for term in ["calculate", "ratio", "backtest"])
        retrieval_required = not lowered.strip().startswith(("what is 2 +", "calculate "))
        reason = "financial research questions need retrieved evidence"
        if tool_required:
            reason = "question asks for analysis that may require a tool"
        return Plan(
            retrieval_required=retrieval_required, tool_required=tool_required, reason=reason
        )
