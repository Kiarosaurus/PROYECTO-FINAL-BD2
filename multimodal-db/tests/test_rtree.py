from __future__ import annotations

from indices.ports import EqualityPredicate, KnnPredicate, SpatialRangePredicate
from indices.rtree import RTreeIndex
from tests.mocks import MockStorageEngine


def test_rtree_searches_exact_point_with_library_index() -> None:
    index = RTreeIndex(column="point")
    records = [
        {"id": 1, "point": (0.0, 0.0)},
        {"id": 2, "point": (4.0, 4.0)},
        {"id": 3, "point": (8.0, 8.0)},
    ]

    result = index.build(records)
    found = index.search(EqualityPredicate(column="point", value=(4.0, 4.0)))

    assert result.success
    assert result.affected == len(records)
    assert found.records == [{"id": 2, "point": (4.0, 4.0)}]


def test_rtree_range_search_returns_points_inside_rectangle() -> None:
    index = RTreeIndex(column="point")
    index.build(
        [
            {"id": 1, "point": (1.0, 1.0)},
            {"id": 2, "point": (2.0, 3.0)},
            {"id": 3, "point": (5.0, 5.0)},
            {"id": 4, "point": (9.0, 9.0)},
        ]
    )

    result = index.search(
        SpatialRangePredicate(
            column="point",
            min_corner=(0.0, 0.0),
            max_corner=(5.0, 5.0),
        )
    )

    assert [record["id"] for record in result.records] == [1, 2, 3]


def test_rtree_knn_returns_nearest_records() -> None:
    index = RTreeIndex(column="point")
    index.build(
        [
            {"id": 1, "point": (10.0, 10.0)},
            {"id": 2, "point": (2.0, 1.0)},
            {"id": 3, "point": (1.0, 1.0)},
            {"id": 4, "point": (7.0, 7.0)},
        ]
    )

    result = index.search(KnnPredicate(column="point", query=(0.0, 0.0), k=2))

    assert [record["id"] for record in result.records] == [3, 2]


def test_rtree_insert_accepts_key_as_fallback_point() -> None:
    index = RTreeIndex(column="point")

    result = index.insert((3.0, 3.0), {"id": 1, "name": "fallback"})
    found = index.search(EqualityPredicate(column="point", value=(3.0, 3.0)))

    assert result.affected == 1
    assert found.records == [{"id": 1, "name": "fallback"}]


def test_rtree_delete_removes_all_records_at_point() -> None:
    index = RTreeIndex(column="point")
    index.build(
        [
            {"id": 1, "point": (1.0, 1.0)},
            {"id": 2, "point": (2.0, 2.0)},
            {"id": 3, "point": (2.0, 2.0)},
            {"id": 4, "point": (3.0, 3.0)},
        ]
    )

    deleted = index.delete((2.0, 2.0))
    result = index.search(
        SpatialRangePredicate(
            column="point",
            min_corner=(0.0, 0.0),
            max_corner=(4.0, 4.0),
        )
    )

    assert deleted.affected == 2
    assert [record["id"] for record in result.records] == [1, 4]


def test_rtree_restores_records_from_mock_storage_snapshot() -> None:
    storage = MockStorageEngine()
    index = RTreeIndex(column="point", storage=storage)
    index.build(
        [
            {"id": 1, "point": (1.0, 1.0)},
            {"id": 2, "point": (2.0, 2.0)},
            {"id": 3, "point": (9.0, 9.0)},
        ]
    )

    restored = RTreeIndex(column="point", storage=storage)
    result = restored.search(
        SpatialRangePredicate(
            column="point",
            min_corner=(0.0, 0.0),
            max_corner=(3.0, 3.0),
        )
    )

    assert storage.stats().disk_writes > 0
    assert [record["id"] for record in result.records] == [1, 2]
