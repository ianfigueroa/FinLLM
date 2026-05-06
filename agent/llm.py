from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from retrieval.citations import Citation
from retrieval.vector_store import SearchResult


class LLMGenerationError(RuntimeError):
    """Raised when an optional external LLM provider cannot produce an answer."""


class PromptLLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class AnswerGenerator(Protocol):
    provider_name: str

    def generate(
        self,
        *,
        question: str,
        results: list[SearchResult],
        citations: list[Citation],
        tool_calls: list[Any],
    ) -> str: ...


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    mode: str
    configured: bool
    reason: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)

    def public_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "configured": self.configured,
            "reason": self.reason,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "api_key_preview": mask_secret(self.api_key),
        }


@dataclass(frozen=True)
class RemoteLLMAnswerGenerator(AnswerGenerator):
    client: PromptLLMClient
    provider_name: str

    def generate(
        self,
        *,
        question: str,
        results: list[SearchResult],
        citations: list[Citation],
        tool_calls: list[Any],
    ) -> str:
        prompt = build_grounded_prompt(question, results, citations, tool_calls)
        answer = self.client.generate(prompt).strip()
        if not answer:
            raise LLMGenerationError("provider returned an empty answer")
        return answer


@dataclass(frozen=True)
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    model: str
    timeout_s: float = 20.0
    transport: httpx.BaseTransport | None = None

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial research assistant. Answer only from the "
                        "provided evidence and preserve citation markers exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
            data = response.json()
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise LLMGenerationError("provider response did not include choices")
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise LLMGenerationError("provider response did not include text content")
            return content
        except httpx.HTTPError as exc:
            raise LLMGenerationError("provider request failed") from exc


@dataclass(frozen=True)
class OllamaClient:
    base_url: str
    model: str
    timeout_s: float = 45.0
    transport: httpx.BaseTransport | None = None

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer financial research questions only from supplied evidence. "
                        "Keep exact citation markers like [C1]."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
            data = response.json()
            message = data.get("message")
            content = message.get("content") if isinstance(message, dict) else data.get("response")
            if not isinstance(content, str):
                raise LLMGenerationError("provider response did not include text content")
            return content
        except httpx.HTTPError as exc:
            raise LLMGenerationError("provider request failed") from exc


def load_llm_settings(env: dict[str, str] | None = None) -> LLMSettings:
    values = os.environ if env is None else env
    provider = values.get("FINLLM_LLM_PROVIDER", "local").strip().lower()
    provider = "openai-compatible" if provider in {"openai", "openai_compatible"} else provider

    if provider == "local":
        return LLMSettings(
            provider="local",
            model="extractive-grounded-answerer",
            mode="extractive",
            configured=True,
            reason="Using deterministic local evidence extraction; no external LLM is required.",
        )

    if provider == "ollama":
        return LLMSettings(
            provider="ollama",
            model=values.get("FINLLM_LLM_MODEL", "qwen2.5:7b-instruct"),
            mode="generative",
            configured=True,
            reason="Configured for a local Ollama model; the Ollama server must be running.",
            base_url=values.get("FINLLM_LLM_BASE_URL", "http://127.0.0.1:11434"),
        )

    if provider == "openai-compatible":
        api_key = values.get("FINLLM_LLM_API_KEY") or values.get("OPENAI_API_KEY")
        configured = bool(api_key)
        return LLMSettings(
            provider="openai-compatible",
            model=values.get("FINLLM_LLM_MODEL", "gpt-4o-mini"),
            mode="generative",
            configured=configured,
            reason=(
                "Configured for an OpenAI-compatible chat completions provider."
                if configured
                else "Set FINLLM_LLM_API_KEY or OPENAI_API_KEY to enable this provider."
            ),
            base_url=values.get("FINLLM_LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key,
        )

    return LLMSettings(
        provider=provider or "unknown",
        model=values.get("FINLLM_LLM_MODEL", ""),
        mode="unavailable",
        configured=False,
        reason="Unsupported FINLLM_LLM_PROVIDER value.",
    )


def build_answer_generator(settings: LLMSettings) -> AnswerGenerator | None:
    if not settings.configured or settings.provider == "local":
        return None
    if settings.provider == "ollama" and settings.base_url:
        return RemoteLLMAnswerGenerator(
            client=OllamaClient(base_url=settings.base_url, model=settings.model),
            provider_name="ollama",
        )
    if settings.provider == "openai-compatible" and settings.base_url and settings.api_key:
        return RemoteLLMAnswerGenerator(
            client=OpenAICompatibleClient(
                base_url=settings.base_url,
                api_key=settings.api_key,
                model=settings.model,
            ),
            provider_name="openai-compatible",
        )
    return None


def openai_compatible_client(
    *,
    base_url: str,
    api_key: str,
    model: str,
    transport: httpx.BaseTransport | None = None,
) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        transport=transport,
    )


def build_grounded_prompt(
    question: str,
    results: list[SearchResult],
    citations: list[Citation],
    tool_calls: list[Any] | None = None,
) -> str:
    lines = [
        "Task: Answer the financial research question using only the supplied evidence.",
        "Policy:",
        "- Never invent citations.",
        "- If evidence is insufficient, say so.",
        "- Separate facts from inference.",
        "- Preserve citation markers exactly, such as [C1].",
        "",
        f"Question: {question}",
        "",
        "Evidence:",
    ]
    for result, citation in zip(results, citations, strict=True):
        metadata = result.chunk.metadata
        section = metadata.section or "Unknown section"
        source = metadata.source_url or metadata.source
        lines.append(f"[{citation.marker}] {metadata.ticker or 'UNKNOWN'} - {section}")
        lines.append(f"Source: {source}")
        lines.append(result.chunk.text.strip())
        lines.append("")

    if tool_calls:
        lines.append("Tool outputs:")
        for call in tool_calls:
            name = getattr(call, "name", "tool")
            ok = getattr(call, "ok", False)
            output = getattr(call, "output", None)
            error = getattr(call, "error", "")
            lines.append(f"- {name}: {output if ok else f'unavailable ({error})'}")
        lines.append("")

    lines.extend(
        [
            "Answer format:",
            "- Start with facts from retrieved evidence.",
            "- Include citations on each factual claim.",
            "- End with limitations and a research-only disclaimer.",
        ]
    )
    return "\n".join(lines)


def mask_secret(secret: str | None) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}...{secret[-4:]}"
