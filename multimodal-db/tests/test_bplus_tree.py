from __future__ import annotations

from indices.bplus_tree import BPlusTreeIndex
from indices.ports import EqualityPredicate, RangePredicate
from tests.mocks import MockStorageEngine


def test_bplus_tree_searches_inserted_records_by_key() -> None:
    storage = MockStorageEngine()
    index = BPlusTreeIndex(column="id", order=4, storage=storage)
    records = [{"id": value, "name": f"row-{value}"} for value in [7, 2, 9, 1, 5, 3]]

    result = index.build(records)

    assert result.success
    assert result.affected == len(records)
    found = index.search(EqualityPredicate(column="id", value=5))
    assert found.records == [{"id": 5, "name": "row-5"}]


def test_bplus_tree_range_scan_walks_linked_leaves_in_order() -> None:
    index = BPlusTreeIndex(column="id", order=4)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 21))

    result = index.search(RangePredicate(column="id", low=6, high=14))

    assert [record["id"] for record in result.records] == list(range(6, 15))


def test_bplus_tree_range_scan_respects_open_bounds_and_limit() -> None:
    index = BPlusTreeIndex(column="id", order=4)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 11))

    predicate = RangePredicate(
        column="id",
        low=3,
        high=8,
        include_low=False,
        include_high=False,
    )
    result = index.search(predicate, k=3)

    assert [record["id"] for record in result.records] == [4, 5, 6]


def test_bplus_tree_keeps_duplicate_keys_clustered() -> None:
    index = BPlusTreeIndex(column="id", order=3)
    index.insert(4, {"id": 4, "name": "first"})
    index.insert(4, {"id": 4, "name": "second"})

    result = index.search(EqualityPredicate(column="id", value=4))

    assert result.records == [
        {"id": 4, "name": "first"},
        {"id": 4, "name": "second"},
    ]


def test_bplus_tree_keeps_duplicates_after_leaf_splits() -> None:
    index = BPlusTreeIndex(column="id", order=3)
    records = [
        {"id": 1, "name": "row-1"},
        {"id": 2, "name": "row-2a"},
        {"id": 3, "name": "row-3"},
        {"id": 4, "name": "row-4"},
        {"id": 5, "name": "row-5"},
        {"id": 2, "name": "row-2b"},
        {"id": 2, "name": "row-2c"},
    ]

    index.build(records)
    index.insert(2, {"id": 2, "name": "row-2d"})
    result = index.search(EqualityPredicate(column="id", value=2))
    range_result = index.search(RangePredicate(column="id", low=2, high=2))

    assert [record["name"] for record in result.records] == [
        "row-2a",
        "row-2b",
        "row-2c",
        "row-2d",
    ]
    assert [record["name"] for record in range_result.records] == [
        "row-2a",
        "row-2b",
        "row-2c",
        "row-2d",
    ]


def test_bplus_tree_leaf_split_position_avoids_duplicate_boundary() -> None:
    index = BPlusTreeIndex(column="id", order=3)

    split = index._leaf_split_position([1, 2, 2, 2, 3])

    assert split == 4


def test_bplus_tree_delete_removes_clustered_key_group() -> None:
    index = BPlusTreeIndex(column="id", order=4)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 8))
    index.insert(4, {"id": 4, "name": "duplicate"})

    deleted = index.delete(4)
    found = index.search(EqualityPredicate(column="id", value=4))
    remaining = index.search(RangePredicate(column="id", low=1, high=7))

    assert deleted.affected == 2
    assert found.records == []
    assert [record["id"] for record in remaining.records] == [1, 2, 3, 5, 6, 7]


def test_bplus_tree_restores_records_from_mock_storage_snapshot() -> None:
    storage = MockStorageEngine()
    index = BPlusTreeIndex(column="id", order=4, storage=storage)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 8))

    restored = BPlusTreeIndex(column="id", order=4, storage=storage)
    result = restored.search(RangePredicate(column="id", low=2, high=5))

    assert storage.stats().disk_writes > 0
    assert [record["id"] for record in result.records] == [2, 3, 4, 5]
