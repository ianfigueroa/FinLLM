from __future__ import annotations

import json

import httpx
import pytest

from ingestion.metadata import DocumentChunk, DocumentMetadata
from retrieval.embeddings import (
    EmbeddingModel,
    HashEmbeddingModel,
    OpenAICompatibleEmbeddingModel,
    build_embedding_model,
    embedding_prefixes,
    load_embedding_settings,
)
from retrieval.vector_store import InMemoryVectorStore

MXBAI_QUERY = "Represent this sentence for searching relevant passages: "


def _recording_model(sent: list[object], **kwargs: object) -> OpenAICompatibleEmbeddingModel:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body["input"])
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(inputs))]}
        )

    return OpenAICompatibleEmbeddingModel(
        base_url="http://ollama.test/v1",
        api_key="ollama",
        model="test-model",
        dimensions=2,
        transport=httpx.MockTransport(handler),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("nomic-embed-text", ("search_query: ", "search_document: ")),
        ("nomic-embed-text:latest", ("search_query: ", "search_document: ")),
        ("mxbai-embed-large", (MXBAI_QUERY, "")),
        ("mxbai-embed-large:latest", (MXBAI_QUERY, "")),
        ("text-embedding-3-small", ("", "")),
    ],
)
def test_embedding_prefixes_by_model_name(model: str, expected: tuple[str, str]) -> None:
    assert embedding_prefixes(model) == expected


def test_hash_model_ignores_role() -> None:
    model = HashEmbeddingModel(dimensions=32)
    text = "Apple total net sales"
    assert model.embed_query(text) == model.embed(text)
    assert model.embed_document(text) == model.embed(text)
    assert model.embed_documents([text, "other"]) == [model.embed(text), model.embed("other")]


def test_query_and_document_prefixes_are_sent() -> None:
    sent: list[object] = []
    model = _recording_model(
        sent, query_prefix="search_query: ", document_prefix="search_document: "
    )

    model.embed_query("revenue")
    model.embed_document("Total net sales 416,161")
    model.embed("raw")

    assert sent == ["search_query: revenue", "search_document: Total net sales 416,161", "raw"]


def test_embed_documents_batches_with_document_prefix() -> None:
    sent: list[object] = []
    model = _recording_model(sent, document_prefix="doc: ", batch_size=2)

    vectors = model.embed_documents(["a", "b", "c"])

    assert sent == [["doc: a", "doc: b"], ["doc: c"]]
    assert vectors == [[1.0, 0.0]] * 3


def test_ollama_settings_pick_model_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINLLM_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("FINLLM_EMBEDDING_MODEL", "mxbai-embed-large")
    monkeypatch.setenv("FINLLM_EMBEDDING_DIMENSIONS", "1024")

    model = build_embedding_model(load_embedding_settings())

    assert isinstance(model, OpenAICompatibleEmbeddingModel)
    assert (model.query_prefix, model.document_prefix) == (MXBAI_QUERY, "")


def test_default_provider_is_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINLLM_EMBEDDING_PROVIDER", raising=False)
    assert isinstance(build_embedding_model(load_embedding_settings()), HashEmbeddingModel)


class _RoleRecorder(EmbeddingModel):
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(("raw", text))
        return [1.0, 0.0]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(("query", text))
        return [1.0, 0.0]

    def embed_document(self, text: str) -> list[float]:
        self.calls.append(("document", text))
        return [1.0, 0.0]


def test_vector_store_embeds_chunks_as_documents_and_queries_as_queries() -> None:
    model = _RoleRecorder()
    store = InMemoryVectorStore(model)
    chunk = DocumentChunk("c1", "chunk text", DocumentMetadata(source="s"), 0, 10)

    store.upsert([chunk])
    store.search("question")

    assert model.calls == [("document", "chunk text"), ("query", "question")]
