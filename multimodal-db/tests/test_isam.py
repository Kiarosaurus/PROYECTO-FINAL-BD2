from __future__ import annotations

from indices.isam import ISAMIndex
from indices.ports import EqualityPredicate, RangePredicate
from tests.mocks import MockBufferManager, MockStorageEngine


def test_isam_builds_static_primary_pages_and_searches_by_key() -> None:
    index = ISAMIndex(column="id", page_capacity=3)
    records = [{"id": value, "name": f"row-{value}"} for value in [8, 1, 5, 2, 3, 7]]

    result = index.build(records)
    found = index.search(EqualityPredicate(column="id", value=5))

    assert result.success
    assert result.affected == len(records)
    assert found.records == [{"id": 5, "name": "row-5"}]


def test_isam_range_scan_returns_records_ordered_across_pages() -> None:
    index = ISAMIndex(column="id", page_capacity=3)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 13))

    result = index.search(RangePredicate(column="id", low=4, high=9))

    assert [record["id"] for record in result.records] == [4, 5, 6, 7, 8, 9]


def test_isam_range_scan_respects_open_bounds_and_limit() -> None:
    index = ISAMIndex(column="id", page_capacity=3)
    index.build({"id": value, "name": f"row-{value}"} for value in range(1, 11))

    predicate = RangePredicate(
        column="id",
        low=2,
        high=8,
        include_low=False,
        include_high=False,
    )
    result = index.search(predicate, k=4)

    assert [record["id"] for record in result.records] == [3, 4, 5, 6]


def test_isam_insert_uses_overflow_when_static_page_is_full() -> None:
    storage = MockStorageEngine()
    buffer = MockBufferManager(storage)
    index = ISAMIndex(
        column="id",
        page_capacity=2,
        overflow_capacity=2,
        buffer=buffer,
    )
    index.build([{"id": 1}, {"id": 2}, {"id": 5}, {"id": 6}])

    index.insert(2, {"id": 2, "name": "overflow-a"})
    index.insert(2, {"id": 2, "name": "overflow-b"})
    result = index.search(EqualityPredicate(column="id", value=2))

    assert index.overflow_page_count() == 1
    assert storage.stats().pages_allocated == 1
    assert [record["id"] for record in result.records] == [2, 2, 2]


def test_isam_chains_multiple_overflow_pages() -> None:
    index = ISAMIndex(column="id", page_capacity=1, overflow_capacity=2)
    index.build([{"id": 10, "name": "primary"}])

    index.insert(10, {"id": 10, "name": "overflow-a"})
    index.insert(10, {"id": 10, "name": "overflow-b"})
    index.insert(10, {"id": 10, "name": "overflow-c"})
    result = index.search(EqualityPredicate(column="id", value=10))

    assert index.overflow_page_count() == 2
    assert [record["name"] for record in result.records] == [
        "primary",
        "overflow-a",
        "overflow-b",
        "overflow-c",
    ]


def test_isam_delete_removes_records_from_primary_and_overflow() -> None:
    index = ISAMIndex(column="id", page_capacity=2, overflow_capacity=2)
    index.build([{"id": 1}, {"id": 2}, {"id": 5}, {"id": 6}])
    index.insert(2, {"id": 2, "name": "overflow"})

    deleted = index.delete(2)
    found = index.search(EqualityPredicate(column="id", value=2))
    remaining = index.search(RangePredicate(column="id", low=1, high=6))

    assert deleted.affected == 2
    assert found.records == []
    assert [record["id"] for record in remaining.records] == [1, 5, 6]


def test_isam_restores_static_pages_and_overflow_from_mock_storage() -> None:
    storage = MockStorageEngine()
    buffer = MockBufferManager(storage)
    index = ISAMIndex(column="id", page_capacity=2, overflow_capacity=2, buffer=buffer)
    index.build([{"id": 1}, {"id": 2}, {"id": 5}, {"id": 6}])
    index.insert(2, {"id": 2, "name": "overflow"})

    restored = ISAMIndex(column="id", page_capacity=2, overflow_capacity=2, buffer=buffer)
    result = restored.search(RangePredicate(column="id", low=1, high=5))

    assert restored.overflow_page_count() == 1
    assert [record["id"] for record in result.records] == [1, 2, 2, 5]
