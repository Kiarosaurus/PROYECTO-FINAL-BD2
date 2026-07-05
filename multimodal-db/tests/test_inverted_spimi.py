from __future__ import annotations

from indices.inverted.spimi_builder import SPIMIBlockBuilder
from indices.inverted.text_index import InvertedIndex
from indices.inverted.text_preprocessor import TextPreprocessor
from indices.ports import EqualityPredicate, TextMatchPredicate
from tests.mocks import MockBufferManager, MockStorageEngine


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


def test_spimi_builder_spills_closed_blocks_to_storage_pages() -> None:
    storage = MockStorageEngine()
    buffer = MockBufferManager(storage)
    builder = SPIMIBlockBuilder(block_document_limit=2, buffer=buffer, file_id="spimi_spill")

    postings = builder.build(
        [
            ("1", "alpha beta beta"),
            ("2", "beta gamma"),
            ("3", "alpha gamma"),
            ("4", "gamma gamma"),
        ]
    )

    assert builder.block_count() == 2
    assert storage.stats().disk_writes > 0
    assert postings["alpha"] == {"1": 1, "3": 1}
    assert postings["beta"] == {"1": 2, "2": 1}
    assert postings["gamma"] == {"2": 1, "3": 1, "4": 2}


def test_spimi_builder_streams_large_blocks_across_multiple_pages() -> None:
    storage = MockStorageEngine()
    buffer = MockBufferManager(storage)
    builder = SPIMIBlockBuilder(block_document_limit=50, buffer=buffer, file_id="spimi_pages")
    documents = [
        (str(doc_id), " ".join(f"w{doc_id}x{term_id}" for term_id in range(40)))
        for doc_id in range(1, 31)
    ]

    postings = builder.build(documents)

    assert builder.block_count() == 1
    assert storage.stats().disk_writes > 1
    assert postings["w10x7"] == {"10": 1}
    assert len(postings) == 30 * 40


def test_text_preprocessor_removes_stopwords_and_stems_terms() -> None:
    preprocessor = TextPreprocessor()

    terms = preprocessor.tokenize("The databases are running and retrieved results")

    assert "the" not in terms
    assert "are" not in terms
    assert "databas" in terms
    assert "run" in terms
    assert "retriev" in terms


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
    assert index.postings_for("systems") == {"10": 1, "20": 1}


def test_inverted_index_uses_preprocessor_for_search_terms() -> None:
    index = InvertedIndex(column="body", block_document_limit=2)
    index.build(
        [
            {"id": 1, "body": "the running database"},
            {"id": 2, "body": "database reports"},
        ]
    )

    result = index.search(TextMatchPredicate(column="body", terms="runs"))

    assert result.records == [{"id": 1, "body": "the running database"}]


def test_inverted_index_text_match_ranks_by_tfidf_cosine() -> None:
    index = InvertedIndex(column="body", block_document_limit=2)
    index.build(
        [
            {"id": 1, "body": "visual search database"},
            {"id": 2, "body": "audio search engine"},
            {"id": 3, "body": "visual retrieval engine"},
        ]
    )

    result = index.search(TextMatchPredicate(column="body", terms="visual search", k=1))
    ranked = index.rank("visual search", k=3)

    assert result.records == [{"id": 1, "body": "visual search database"}]
    assert ranked[0][0] == "1"
    assert ranked[0][1] > ranked[1][1]


def test_inverted_index_accepts_plain_string_query() -> None:
    index = InvertedIndex(column="body", block_document_limit=2)
    index.build(
        [
            {"id": 1, "body": "visual search database"},
            {"id": 2, "body": "audio retrieval engine"},
        ]
    )

    result = index.search("visual")

    assert result.success
    assert result.records == [{"id": 1, "body": "visual search database"}]


def test_inverted_index_rejects_foreign_predicate() -> None:
    index = InvertedIndex(column="body", block_document_limit=2)
    index.build([{"id": 1, "body": "alpha beta"}])

    result = index.search(EqualityPredicate(column="body", value="alpha"))

    assert not result.success
    assert result.records == []
    assert "EqualityPredicate" in result.message


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


def test_inverted_index_computes_document_norms() -> None:
    index = InvertedIndex(column="body", block_document_limit=2)

    index.build(
        [
            {"id": 1, "body": "alpha alpha beta"},
            {"id": 2, "body": "beta gamma"},
        ]
    )

    assert index.document_norm(1) > index.document_norm(2)
    assert index.document_norm("missing") == 0.0


def test_inverted_index_restores_snapshot_from_mock_storage() -> None:
    storage = MockStorageEngine()
    buffer = MockBufferManager(storage)
    index = InvertedIndex(column="body", block_document_limit=2, buffer=buffer)
    index.build(
        [
            {"id": 1, "body": "alpha beta"},
            {"id": 2, "body": "beta gamma"},
        ]
    )

    restored = InvertedIndex(column="body", block_document_limit=2, buffer=buffer)
    result = restored.search(TextMatchPredicate(column="body", terms="beta"))

    assert storage.stats().disk_writes > 0
    assert [record["id"] for record in result.records] == [1, 2]
    assert restored.document_norm(1) > 0.0


def test_inverted_index_streams_postings_across_storage_pages() -> None:
    storage = MockStorageEngine()
    buffer = MockBufferManager(storage)
    index = InvertedIndex(column="body", block_document_limit=10, buffer=buffer)
    records = [
        {
            "id": doc_id,
            "body": " ".join(f"term{doc_id}_{term_id}" for term_id in range(40)),
        }
        for doc_id in range(1, 30)
    ]

    index.build(records)
    restored = InvertedIndex(column="body", block_document_limit=10, buffer=buffer)
    result = restored.search(TextMatchPredicate(column="body", terms="term10_7"))

    assert index.posting_page_count() > 1
    assert restored.posting_page_count() == index.posting_page_count()
    assert result.records == [records[9]]
