from __future__ import annotations

from indices.inverted.spimi_builder import SPIMIBlockBuilder
from indices.inverted.text_index import InvertedIndex
from indices.ports import TextMatchPredicate
from tests.mocks import MockStorageEngine


def test_spimi_builder_flushes_blocks_and_merges_terms_with_heap() -> None:
    builder = SPIMIBlockBuilder(block_document_limit=2)

    postings = builder.build(
        [
            ("1", "alpha beta beta"),
            ("2", "beta gamma"),
            ("3", "alpha gamma"),
            ("4", "gamma gamma"),
        ]
    )

    assert builder.block_count() == 2
    assert postings["alpha"] == {"1": 1, "3": 1}
    assert postings["beta"] == {"1": 2, "2": 1}
    assert postings["gamma"] == {"2": 1, "3": 1, "4": 2}


def test_inverted_index_builds_spimi_postings_for_text_records() -> None:
    index = InvertedIndex(column="body", block_document_limit=2)
    records = [
        {"id": 10, "body": "database systems database"},
        {"id": 20, "body": "systems retrieval"},
        {"id": 30, "body": "database retrieval"},
    ]

    result = index.build(records)

    assert result.success
    assert result.affected == len(records)
    assert index.block_count() == 2
    assert index.postings_for("database") == {"10": 2, "30": 1}


def test_inverted_index_text_match_requires_all_query_terms() -> None:
    index = InvertedIndex(column="body", block_document_limit=2)
    index.build(
        [
            {"id": 1, "body": "visual search database"},
            {"id": 2, "body": "audio search engine"},
            {"id": 3, "body": "visual retrieval engine"},
        ]
    )

    result = index.search(TextMatchPredicate(column="body", terms="visual search"))

    assert result.records == [{"id": 1, "body": "visual search database"}]


def test_inverted_index_insert_and_delete_update_postings() -> None:
    index = InvertedIndex(column="body", block_document_limit=2)
    index.build([{"id": 1, "body": "alpha beta"}])

    index.insert(2, {"id": 2, "body": "alpha gamma"})
    inserted = index.search(TextMatchPredicate(column="body", terms="alpha"))
    deleted = index.delete(1)
    remaining = index.search(TextMatchPredicate(column="body", terms="alpha"))

    assert [record["id"] for record in inserted.records] == [1, 2]
    assert deleted.affected == 1
    assert remaining.records == [{"id": 2, "body": "alpha gamma"}]
    assert index.postings_for("beta") == {}


def test_inverted_index_restores_snapshot_from_mock_storage() -> None:
    storage = MockStorageEngine()
    index = InvertedIndex(column="body", block_document_limit=2, storage=storage)
    index.build(
        [
            {"id": 1, "body": "alpha beta"},
            {"id": 2, "body": "beta gamma"},
        ]
    )

    restored = InvertedIndex(column="body", block_document_limit=2, storage=storage)
    result = restored.search(TextMatchPredicate(column="body", terms="beta"))

    assert storage.stats().disk_writes > 0
    assert [record["id"] for record in result.records] == [1, 2]
