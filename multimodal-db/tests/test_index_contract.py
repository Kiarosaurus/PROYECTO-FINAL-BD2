from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest

from core.metrics import OperationResult
from core.ports.index import Index
from indices.bplus_tree import BPlusTreeIndex
from indices.extendible_hash import ExtendibleHashIndex
from indices.inverted.text_index import InvertedIndex
from indices.isam import ISAMIndex
from indices.ports import (
    EqualityPredicate,
    KnnPredicate,
    RangePredicate,
    SpatialRangePredicate,
    TextMatchPredicate,
)
from indices.rtree import RTreeIndex
from tests.mocks import MockStorageEngine


@dataclass(frozen=True)
class IndexContractCase:
    name: str
    factory: Callable[[MockStorageEngine | None], Index]
    records: list[Any]
    insert_key: Any
    insert_record: Any
    search_predicate: Any
    search_expected_ids: list[Any]
    limited_predicate: Any
    limited_expected_count: int
    delete_key: Any
    deleted_id: Any
    deleted_predicate: Any
    id_field: str = "id"


CONTRACT_CASES = [
    IndexContractCase(
        name="bplus",
        factory=lambda storage=None: BPlusTreeIndex(column="id", order=4, storage=storage),
        records=[{"id": value, "name": f"row-{value}"} for value in [1, 2, 3, 4]],
        insert_key=5,
        insert_record={"id": 5, "name": "row-5"},
        search_predicate=RangePredicate(column="id", low=2, high=4),
        search_expected_ids=[2, 3, 4],
        limited_predicate=RangePredicate(column="id", low=1, high=5),
        limited_expected_count=2,
        delete_key=3,
        deleted_id=3,
        deleted_predicate=EqualityPredicate(column="id", value=3),
    ),
    IndexContractCase(
        name="isam",
        factory=lambda storage=None: ISAMIndex(column="id", page_capacity=2, storage=storage),
        records=[{"id": value, "name": f"row-{value}"} for value in [1, 2, 3, 4]],
        insert_key=5,
        insert_record={"id": 5, "name": "row-5"},
        search_predicate=RangePredicate(column="id", low=2, high=4),
        search_expected_ids=[2, 3, 4],
        limited_predicate=RangePredicate(column="id", low=1, high=5),
        limited_expected_count=2,
        delete_key=3,
        deleted_id=3,
        deleted_predicate=EqualityPredicate(column="id", value=3),
    ),
    IndexContractCase(
        name="hash",
        factory=lambda storage=None: ExtendibleHashIndex(column="id", bucket_capacity=2, storage=storage),
        records=[{"id": value, "name": f"row-{value}"} for value in [1, 2, 3, 4]],
        insert_key=5,
        insert_record={"id": 5, "name": "row-5"},
        search_predicate=EqualityPredicate(column="id", value=3),
        search_expected_ids=[3],
        limited_predicate=RangePredicate(column="id", low=1, high=5),
        limited_expected_count=2,
        delete_key=3,
        deleted_id=3,
        deleted_predicate=EqualityPredicate(column="id", value=3),
    ),
    IndexContractCase(
        name="rtree",
        factory=lambda storage=None: RTreeIndex(column="point", storage=storage),
        records=[
            {"id": 1, "point": (1.0, 1.0)},
            {"id": 2, "point": (2.0, 2.0)},
            {"id": 3, "point": (3.0, 3.0)},
            {"id": 4, "point": (9.0, 9.0)},
        ],
        insert_key=(4.0, 4.0),
        insert_record={"id": 5, "point": (4.0, 4.0)},
        search_predicate=SpatialRangePredicate(
            column="point",
            min_corner=(1.5, 1.5),
            max_corner=(3.5, 3.5),
        ),
        search_expected_ids=[2, 3],
        limited_predicate=KnnPredicate(column="point", query=(0.0, 0.0), k=4),
        limited_expected_count=2,
        delete_key=(3.0, 3.0),
        deleted_id=3,
        deleted_predicate=EqualityPredicate(column="point", value=(3.0, 3.0)),
    ),
    IndexContractCase(
        name="inverted",
        factory=lambda storage=None: InvertedIndex(column="body", block_document_limit=2, storage=storage),
        records=[
            {"id": 1, "body": "visual database search"},
            {"id": 2, "body": "audio retrieval engine"},
            {"id": 3, "body": "visual retrieval engine"},
            {"id": 4, "body": "database storage engine"},
        ],
        insert_key=5,
        insert_record={"id": 5, "body": "visual search engine"},
        search_predicate=TextMatchPredicate(column="body", terms="visual retrieval"),
        search_expected_ids=[3, 2, 5, 1],
        limited_predicate=TextMatchPredicate(column="body", terms="engine", k=4),
        limited_expected_count=2,
        delete_key=3,
        deleted_id=3,
        deleted_predicate=TextMatchPredicate(column="body", terms="visual retrieval"),
    ),
]


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=[case.name for case in CONTRACT_CASES])
def test_index_contract_build_insert_search_and_delete(case: IndexContractCase) -> None:
    index = case.factory(None)

    build_result = index.build(case.records)
    insert_result = index.insert(case.insert_key, case.insert_record)
    search_result = index.search(case.search_predicate)
    limited_result = index.search(case.limited_predicate, k=case.limited_expected_count)
    delete_result = index.delete(case.delete_key)
    deleted_result = index.search(case.deleted_predicate)

    assert isinstance(index, Index)
    assert isinstance(build_result, OperationResult)
    assert isinstance(insert_result, OperationResult)
    assert isinstance(search_result, OperationResult)
    assert build_result.success
    assert build_result.affected == len(case.records)
    assert insert_result.success
    assert insert_result.affected == 1
    assert _ids(search_result.records, case.id_field) == case.search_expected_ids
    assert len(limited_result.records) == case.limited_expected_count
    assert delete_result.affected >= 1
    assert case.deleted_id not in _ids(deleted_result.records, case.id_field)


@pytest.mark.parametrize("case", CONTRACT_CASES, ids=[case.name for case in CONTRACT_CASES])
def test_index_contract_restores_from_mock_storage(case: IndexContractCase) -> None:
    storage = MockStorageEngine()
    index = case.factory(storage)

    build_result = index.build(case.records)
    index.insert(case.insert_key, case.insert_record)
    restored = case.factory(storage)
    search_result = restored.search(case.search_predicate)

    assert build_result.success
    assert storage.stats().disk_writes > 0
    assert _ids(search_result.records, case.id_field) == case.search_expected_ids


def _ids(records: list[Any], field: str) -> list[Any]:
    return [record[field] if isinstance(record, dict) else getattr(record, field) for record in records]
