from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    mode: str = Field(default="basic_rag", pattern="^(basic_rag|rag_rerank|self_verify)$")
    filters: dict[str, str] | None = None


class SecUrlIngestionRequest(BaseModel):
    url: str = Field(min_length=20, max_length=1_000)
    ticker: str = Field(min_length=1, max_length=12)
    company: str = Field(min_length=1, max_length=120)
    form_type: str = Field(min_length=1, max_length=32)
    filing_date: str = Field(min_length=4, max_length=32)


class ApiResponse(BaseModel):
    data: Any
