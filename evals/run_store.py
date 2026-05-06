from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EvalRunStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        run = {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
            **payload,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run, ensure_ascii=False) + "\n")
        return run

    def recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0 or not self._path.exists():
            return []
        runs = self._read_all()
        return list(reversed(runs[-limit:]))

    def summary(self) -> dict[str, Any]:
        runs = self._read_all()
        if not runs:
            return {
                "runs": 0,
                "latest_best_mode": "",
                "latest_quality_score": 0.0,
                "average_regression_pass_rate": 0.0,
            }
        latest = runs[-1]
        return {
            "runs": len(runs),
            "latest_best_mode": latest.get("best_mode", ""),
            "latest_quality_score": _latest_quality_score(latest),
            "average_regression_pass_rate": round(
                sum(float(run.get("regression_pass_rate", 0.0)) for run in runs) / len(runs),
                4,
            ),
        }

    def _read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        runs: list[dict[str, Any]] = []
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    runs.append(json.loads(line))
        return runs


def _latest_quality_score(run: dict[str, Any]) -> float:
    best_mode = run.get("best_mode")
    mode_results = run.get("mode_results", [])
    if not isinstance(mode_results, list):
        return 0.0
    for result in mode_results:
        if isinstance(result, dict) and result.get("mode") == best_mode:
            return float(result.get("quality_score", 0.0))
    return 0.0
