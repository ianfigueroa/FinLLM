from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QueryMetric:
    query_id: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass
class CostLatencyTracker:
    records: list[QueryMetric] = field(default_factory=list)

    def record(
        self,
        *,
        query_id: str,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        self.records.append(
            QueryMetric(query_id, latency_ms, prompt_tokens, completion_tokens, cost_usd)
        )

    def summary(self) -> dict[str, float | int]:
        if not self.records:
            return {"queries": 0, "avg_latency_ms": 0.0, "total_cost_usd": 0.0}
        return {
            "queries": len(self.records),
            "avg_latency_ms": round(
                sum(record.latency_ms for record in self.records) / len(self.records), 4
            ),
            "total_cost_usd": round(sum(record.cost_usd for record in self.records), 6),
        }
