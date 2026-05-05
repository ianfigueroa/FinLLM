from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_SECRET_TOKENS = ("key", "secret", "token", "password")


class StructuredLogger:
    def __init__(self, name: str = "finllm") -> None:
        self._logger = logging.getLogger(name)

    def event(self, event: str, **context: Any) -> dict[str, Any]:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **redact_context(context),
        }
        self._logger.info(json.dumps(payload, sort_keys=True))
        return payload


def redact_context(context: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in context.items():
        if any(token in key.lower() for token in _SECRET_TOKENS):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted
