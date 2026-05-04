from __future__ import annotations

import re

_SEC_NOISE_PATTERNS = [
    re.compile(r"^\s*table of contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
]


def clean_text(text: str) -> str:
    """Normalize filing text without destroying section boundaries."""
    cleaned_lines: list[str] = []

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            continue
        if any(pattern.match(line) for pattern in _SEC_NOISE_PATTERNS):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

