from __future__ import annotations

SYSTEM_MESSAGE = (
    "Answer financial research questions using only supplied evidence and exact citations."
)


def make_lora_records(raft_examples: list[dict[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for example in raft_examples:
        context = _format_context(example)
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": f"{context}\n\nQuestion: {example['question']}"},
                    {"role": "assistant", "content": str(example["answer"])},
                ]
            }
        )
    return records


def _format_context(example: dict[str, object]) -> str:
    chunks = list(example.get("relevant_chunks", [])) + list(example.get("distractor_chunks", []))
    lines = ["Evidence:"]
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"[C{index}] {chunk['text']}")
    return "\n".join(lines)
