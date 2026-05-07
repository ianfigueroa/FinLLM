from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryTurn:
    question: str
    answer: str
    mode: str
    citation_markers: list[str]


class ConversationMemory:
    def __init__(self, *, max_turns: int = 20) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self._max_turns = max_turns
        self._turns: list[MemoryTurn] = []

    def append(self, turn: MemoryTurn) -> None:
        self._turns.append(turn)
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def recent(self, limit: int = 5) -> list[MemoryTurn]:
        if limit <= 0:
            return []
        return self._turns[-limit:]

    def clear(self) -> None:
        self._turns.clear()
