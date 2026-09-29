from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass

import httpx

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")

# Instruction prefixes the model cards ask for; retrieval quality drops without them.
_MODEL_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", ""),
}


class EmbeddingModel:
    dimensions: int

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_document(text) for text in texts]


class EmbeddingProviderError(RuntimeError):
    """Raised when a configured remote embedding provider cannot produce a vector."""


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str
    model: str
    dimensions: int
    configured: bool
    reason: str
    base_url: str | None = None
    api_key: str | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "configured": self.configured,
            "reason": self.reason,
            "base_url": self.base_url,
            "api_key_present": bool(self.api_key),
            "api_key_preview": _mask_secret(self.api_key),
        }


@dataclass(frozen=True)
class HashEmbeddingModel(EmbeddingModel):
    """Deterministic local embeddings for tests and offline demos."""

    dimensions: int = 256

    def embed(self, text: str) -> list[float]:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")

        vector = [0.0 for _ in range(self.dimensions)]
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        return normalize(vector)


@dataclass(frozen=True)
class OpenAICompatibleEmbeddingModel(EmbeddingModel):
    base_url: str
    api_key: str
    model: str
    dimensions: int = 1536
    timeout_s: float = 20.0
    transport: httpx.BaseTransport | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    batch_size: int = 32

    def embed(self, text: str) -> list[float]:
        return self._request(text)[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(f"{self.query_prefix}{text}")

    def embed_document(self, text: str) -> list[float]:
        return self.embed(f"{self.document_prefix}{text}")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [f"{self.document_prefix}{text}" for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(prefixed), self.batch_size):
            vectors.extend(self._request(prefixed[start : start + self.batch_size]))
        return vectors

    def _request(self, inputs: str | list[str]) -> list[list[float]]:
        expected = len(inputs) if isinstance(inputs, list) else 1
        payload = {"model": self.model, "input": inputs}
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s) as client:
                response = client.post(
                    f"{self.base_url.rstrip('/')}/embeddings",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
            data = response.json()
            records = data.get("data")
            if not isinstance(records, list) or len(records) != expected:
                raise EmbeddingProviderError("embedding response did not include data")
            if not all(isinstance(record, dict) for record in records):
                raise EmbeddingProviderError("embedding response did not include a vector")
            vectors = []
            for record in sorted(records, key=lambda record: record.get("index", 0)):
                embedding = record.get("embedding")
                if not isinstance(embedding, list):
                    raise EmbeddingProviderError("embedding response did not include a vector")
                vector = [float(value) for value in embedding]
                if self.dimensions and len(vector) != self.dimensions:
                    raise EmbeddingProviderError(
                        "embedding vector dimensions did not match settings"
                    )
                vectors.append(normalize(vector))
            return vectors
        except httpx.HTTPError as exc:
            raise EmbeddingProviderError("embedding provider request failed") from exc
        except (TypeError, ValueError) as exc:
            raise EmbeddingProviderError("embedding provider returned an invalid vector") from exc


def load_embedding_settings() -> EmbeddingSettings:
    provider = os.getenv("FINLLM_EMBEDDING_PROVIDER", "hash").strip().lower()
    provider = "openai-compatible" if provider in {"openai", "openai_compatible"} else provider
    dimensions = _int_env("FINLLM_EMBEDDING_DIMENSIONS", 256)

    if provider == "hash":
        return EmbeddingSettings(
            provider="hash",
            model="hash-embedding-256",
            dimensions=dimensions,
            configured=True,
            reason="Using deterministic local hash embeddings.",
        )

    if provider == "ollama":
        return EmbeddingSettings(
            provider="ollama",
            model=os.getenv("FINLLM_EMBEDDING_MODEL", "nomic-embed-text"),
            dimensions=_int_env("FINLLM_EMBEDDING_DIMENSIONS", 768),
            configured=True,
            reason=(
                "Configured for a local Ollama embedding model; the Ollama server must be running."
            ),
            base_url=os.getenv("FINLLM_EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1"),
            api_key="ollama",
        )

    if provider == "openai-compatible":
        api_key = os.getenv("FINLLM_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        configured = bool(api_key)
        return EmbeddingSettings(
            provider="openai-compatible",
            model=os.getenv("FINLLM_EMBEDDING_MODEL", "text-embedding-3-small"),
            dimensions=_int_env("FINLLM_EMBEDDING_DIMENSIONS", 1536),
            configured=configured,
            reason=(
                "Configured for an OpenAI-compatible embeddings endpoint."
                if configured
                else "Set FINLLM_EMBEDDING_API_KEY or OPENAI_API_KEY to enable this provider."
            ),
            base_url=os.getenv("FINLLM_EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key,
        )

    return EmbeddingSettings(
        provider=provider or "unknown",
        model=os.getenv("FINLLM_EMBEDDING_MODEL", ""),
        dimensions=dimensions,
        configured=False,
        reason="Unsupported FINLLM_EMBEDDING_PROVIDER value.",
    )


def build_embedding_model(settings: EmbeddingSettings) -> EmbeddingModel:
    # Ollama exposes an OpenAI-compatible /v1/embeddings endpoint, so it reuses the
    # same client. It needs no real key; the placeholder is ignored by the server.
    if settings.provider in {"openai-compatible", "ollama"} and settings.configured:
        if settings.base_url is None or settings.api_key is None:
            raise ValueError(f"{settings.provider} embeddings require base_url and api_key")
        query_prefix, document_prefix = embedding_prefixes(settings.model)
        return OpenAICompatibleEmbeddingModel(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            dimensions=settings.dimensions,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )
    return HashEmbeddingModel(dimensions=settings.dimensions)


def embedding_prefixes(model: str) -> tuple[str, str]:
    """Return the (query, document) prefixes a model expects, ignoring any ":tag"."""
    return _MODEL_PREFIXES.get(model.split(":", 1)[0], ("", ""))


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _mask_secret(secret: str | None) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "****"
    return f"{secret[:4]}...{secret[-4:]}"
