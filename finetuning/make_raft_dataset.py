from __future__ import annotations

from ingestion.metadata import DocumentChunk


def make_raft_examples(
    chunks: list[DocumentChunk],
    *,
    questions_per_chunk: int = 1,
    distractor_count: int = 2,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for chunk in chunks:
        for question_index in range(questions_per_chunk):
            distractors = [
                _chunk_payload(candidate)
                for candidate in chunks
                if candidate.chunk_id != chunk.chunk_id
            ][:distractor_count]
            examples.append(
                {
                    "question": _question_for_chunk(chunk, question_index),
                    "relevant_chunks": [_chunk_payload(chunk)],
                    "distractor_chunks": distractors,
                    "answer": f"{_first_sentence(chunk.text)} [C1].",
                    "metadata": {
                        "ticker": chunk.metadata.ticker,
                        "section": chunk.metadata.section,
                        "source": chunk.metadata.source,
                    },
                    "example_id": f"raft-{chunk.chunk_id}-{question_index}",
                }
            )
    return examples


def _question_for_chunk(chunk: DocumentChunk, question_index: int) -> str:
    section = chunk.metadata.section or "the document"
    ticker = chunk.metadata.ticker or "the company"
    if question_index % 2 == 0:
        return f"What does {ticker} disclose in {section}?"
    return f"Summarize the evidence from {section} for {ticker}."


def _chunk_payload(chunk: DocumentChunk) -> dict[str, str]:
    return {"chunk_id": chunk.chunk_id, "text": chunk.text, "section": chunk.metadata.section or ""}


def _first_sentence(text: str) -> str:
    return text.replace("\n", " ").split(".")[0].strip()
