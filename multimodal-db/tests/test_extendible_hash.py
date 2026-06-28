from __future__ import annotations

from indices.extendible_hash import ExtendibleHashIndex
from indices.ports import EqualityPredicate, RangePredicate
from tests.mocks import MockStorageEngine


def test_extendible_hash_searches_inserted_records_by_key() -> None:
    index = ExtendibleHashIndex(column="id", bucket_capacity=2)
    records = [{"id": value, "name": f"row-{value}"} for value in [5, 1, 9, 3]]

    result = index.build(records)
    found = index.search(EqualityPredicate(column="id", value=9))

    assert result.success
    assert result.affected == len(records)
    assert found.records == [{"id": 9, "name": "row-9"}]


def test_extendible_hash_splits_buckets_and_grows_directory() -> None:
    storage = MockStorageEngine()
    index = ExtendibleHashIndex(column="id", bucket_capacity=1, storage=storage)

    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 9))

    assert index.bucket_count() > 2
    assert index.directory_size() > 2
    assert storage.stats().pages_allocated > 0
    for value in range(1, 9):
        result = index.search(EqualityPredicate(column="id", value=value))
        assert result.records == [{"id": value, "name": f"row-{value}"}]


def test_extendible_hash_keeps_duplicate_keys_in_same_bucket() -> None:
    index = ExtendibleHashIndex(column="id", bucket_capacity=1)

    index.insert(7, {"id": 7, "name": "first"})
    index.insert(7, {"id": 7, "name": "second"})
    result = index.search(EqualityPredicate(column="id", value=7))

    assert result.records == [
        {"id": 7, "name": "first"},
        {"id": 7, "name": "second"},
    ]
    assert index.bucket_count() == 2


def test_extendible_hash_range_scan_checks_all_unique_buckets() -> None:
    index = ExtendibleHashIndex(column="id", bucket_capacity=2)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 13))

    result = index.search(RangePredicate(column="id", low=4, high=9))

    assert [record["id"] for record in result.records] == [4, 5, 6, 7, 8, 9]


def test_extendible_hash_range_scan_respects_open_bounds_and_limit() -> None:
    index = ExtendibleHashIndex(column="id", bucket_capacity=2)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 11))

    predicate = RangePredicate(
        column="id",
        low=2,
        high=8,
        include_low=False,
        include_high=False,
    )
    result = index.search(predicate, k=3)

    assert [record["id"] for record in result.records] == [3, 4, 5]


def test_extendible_hash_delete_removes_all_records_for_key() -> None:
    index = ExtendibleHashIndex(column="id", bucket_capacity=2)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 7))
    index.insert(4, {"id": 4, "name": "duplicate"})

    deleted = index.delete(4)
    found = index.search(EqualityPredicate(column="id", value=4))
    remaining = index.search(RangePredicate(column="id", low=1, high=6))

    assert deleted.affected == 2
    assert found.records == []
    assert [record["id"] for record in remaining.records] == [1, 2, 3, 5, 6]


def test_extendible_hash_restores_directory_from_mock_storage() -> None:
    storage = MockStorageEngine()
    index = ExtendibleHashIndex(column="id", bucket_capacity=1, storage=storage)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 7))

    restored = ExtendibleHashIndex(column="id", bucket_capacity=1, storage=storage)
    result = restored.search(RangePredicate(column="id", low=2, high=5))

    assert restored.bucket_count() == index.bucket_count()
    assert restored.directory_size() == index.directory_size()
    assert [record["id"] for record in result.records] == [2, 3, 4, 5]
