from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestion.metadata import DocumentChunk, DocumentMetadata
from retrieval.embeddings import EmbeddingModel
from retrieval.vector_store import SearchResult


class ChromaVectorStore:
    """Persistent Chroma adapter for local vector database deployments."""

    def __init__(
        self,
        *,
        path: str | Path,
        embedding_model: EmbeddingModel,
        collection_name: str = "financial_chunks",
    ) -> None:
        import chromadb

        self._embedding_model = embedding_model
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[self._embedding_model.embed(chunk.text) for chunk in chunks],
            metadatas=[_metadata_to_chroma(chunk) for chunk in chunks],
        )

    def search(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        limit: int = 8,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        query_result = self._collection.query(
            query_embeddings=[self._embedding_model.embed(query)],
            n_results=limit,
            where=filters or None,
        )
        ids = query_result.get("ids", [[]])[0]
        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True),
            start=1,
        ):
            chunk = _metadata_from_chroma(chunk_id, text, metadata)
            results.append(
                SearchResult(chunk=chunk, score=max(0.0, 1.0 - float(distance)), rank=rank)
            )
        return results


def _metadata_to_chroma(chunk: DocumentChunk) -> dict[str, str | int]:
    metadata = chunk.metadata
    payload: dict[str, str | int] = {
        "source": metadata.source,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }
    for key in ["ticker", "company", "form_type", "filing_date", "section", "source_url"]:
        value = getattr(metadata, key)
        if value is not None:
            payload[key] = value
    return payload


def _metadata_from_chroma(chunk_id: str, text: str, metadata: dict[str, Any]) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        metadata=DocumentMetadata(
            ticker=metadata.get("ticker"),
            company=metadata.get("company"),
            form_type=metadata.get("form_type"),
            filing_date=metadata.get("filing_date"),
            section=metadata.get("section"),
            source=metadata["source"],
            source_url=metadata.get("source_url"),
        ),
        start_char=int(metadata["start_char"]),
        end_char=int(metadata["end_char"]),
    )
