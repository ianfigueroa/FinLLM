from __future__ import annotations

import hashlib
import random

from evals.real_corpus_eval import (
    ADDED_CASE_IDS,
    MORE_FIN_FACTS,
    ORIGINAL_CASE_IDS,
    TEST_CASE_IDS,
    _split_key,
    build_cases,
    split_of,
)
from ingestion.metadata import DocumentChunk, DocumentMetadata

# Changing a question id or the split rule changes this; do it only on purpose.
PINNED_TEST_SPLIT_SHA256 = "e62012e265197067df40d63d36c08259809d61c7713580259537d7304c5205b4"


def test_split_is_pinned() -> None:
    digest = hashlib.sha256("\n".join(sorted(TEST_CASE_IDS)).encode()).hexdigest()
    assert digest == PINNED_TEST_SPLIT_SHA256


def test_split_does_not_depend_on_question_order() -> None:
    shuffled = list(ADDED_CASE_IDS)
    random.Random(7).shuffle(shuffled)
    assert frozenset(sorted(shuffled, key=_split_key)[1::2]) == TEST_CASE_IDS


def test_original_questions_are_all_dev_and_added_split_in_half() -> None:
    assert len(ORIGINAL_CASE_IDS) == 34
    assert len(set(ADDED_CASE_IDS)) == len(ADDED_CASE_IDS)
    assert not ORIGINAL_CASE_IDS & set(ADDED_CASE_IDS)
    assert all(split_of(case_id) == "dev" for case_id in ORIGINAL_CASE_IDS)
    assert TEST_CASE_IDS.issubset(ADDED_CASE_IDS)
    assert len(TEST_CASE_IDS) == len(ADDED_CASE_IDS) // 2


def test_figures_never_appear_in_questions() -> None:
    assert all(not any(ch.isdigit() for ch in metric) for _, metric, _ in MORE_FIN_FACTS)


def test_gold_matches_whole_figures_in_either_form() -> None:
    def chunk(chunk_id: str, text: str) -> DocumentChunk:
        return DocumentChunk(chunk_id, text, DocumentMetadata(source="s", ticker="AAPL"), 0, 0)

    chunks = [
        chunk("statement", "Research and development 34,550 31,370 29,915"),
        chunk("xbrl", "Research and development 2025 34550000000"),
        chunk("longer-number", "Other 134,550 and 34,5501"),
    ]
    cases = {case.case_id: case for case in build_cases(chunks, {"AAPL"})}

    gold = cases["aapl-research-and-development-expense"].expected_chunk_ids
    assert gold == ["statement", "xbrl"]
