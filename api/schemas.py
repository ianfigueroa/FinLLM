from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    mode: str = Field(default="basic_rag", pattern="^(basic_rag|rag_rerank|self_verify)$")
    filters: dict[str, str] | None = None


class SecUrlIngestionRequest(BaseModel):
    url: str = Field(min_length=20, max_length=1_000)
    ticker: str = Field(default="", max_length=12)
    company: str = Field(default="", max_length=120)
    form_type: str = Field(default="", max_length=32)
    filing_date: str = Field(default="", max_length=32)


class SecUrlMetadataRequest(BaseModel):
    url: str = Field(min_length=20, max_length=1_000)


class RaftExperimentRequest(BaseModel):
    max_examples: int = Field(default=20, ge=1, le=200)
    distractor_count: int = Field(default=2, ge=0, le=5)
    base_model: str = Field(default="local-sim", min_length=1, max_length=80)


class ApiResponse(BaseModel):
    data: Any
