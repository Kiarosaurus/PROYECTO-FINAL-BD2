from __future__ import annotations

import json
import math
from typing import Any, Iterable

from rtree import index as rtree_index

from core.metrics import IOStats, OperationResult
from core.ports.buffer import BufferManager
from core.ports.index import Index
from indices.ports import (
    EqualityPredicate,
    KnnPredicate,
    SearchPredicate,
    SpatialRangePredicate,
)


class RTreeIndex(Index):

    def __init__(
        self,
        column: str,
        dimensions: int = 2,
        max_entries: int = 50,
        buffer: BufferManager | None = None,
        file_id: str | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("R-Tree dimensions must be positive")
        if max_entries < 4:
            raise ValueError("R-Tree max entries must be at least 4")
        self.column = column
        self.dimensions = dimensions
        self.max_entries = max_entries
        self.buffer = buffer
        self.file_id = file_id or f"rtree_{column}"
        self._records: dict[int, Any] = {}
        self._points: dict[int, tuple[float, ...]] = {}
        self._next_id = 1
        self._index = self._new_index()
        self._load_snapshot()

    def build(self, records: Iterable[Any]) -> OperationResult:
        self._records = {}
        self._points = {}
        self._next_id = 1
        self._index = self._new_index()
        affected = 0
        for record in records:
            self._insert_record(record)
            affected += 1
        self._persist_snapshot()
        return OperationResult(affected=affected, io=self._stats())

    def insert(self, key: Any, record: Any) -> OperationResult:
        self._insert_record(record, fallback_point=key)
        self._persist_snapshot()
        return OperationResult(affected=1, io=self._stats())

    def search(self, predicate: SearchPredicate | Any, k: int | None = None) -> OperationResult:
        if isinstance(predicate, SpatialRangePredicate):
            records = self._range_search(predicate.min_corner, predicate.max_corner)
        elif isinstance(predicate, KnnPredicate):
            records = self._knn_search(predicate.query, predicate.k)
        elif isinstance(predicate, EqualityPredicate):
            records = self._point_search(predicate.value)
        else:
            records = self._point_search(predicate)
        if k is not None:
            records = records[:k]
        return OperationResult(records=records, io=self._stats())

    def delete(self, key: Any) -> OperationResult:
        point = self._normalize_point(key)
        removed = 0
        for record_id in list(self._index.intersection(self._point_mbr(point))):
            stored = self._points.get(record_id)
            if stored != point:
                continue
            self._index.delete(record_id, self._point_mbr(stored))
            self._records.pop(record_id, None)
            self._points.pop(record_id, None)
            removed += 1
        if removed:
            self._persist_snapshot()
        return OperationResult(affected=removed, io=self._stats())

    def size(self) -> int:
        return len(self._records)

    def _new_index(self) -> rtree_index.Index:
        properties = rtree_index.Property()
        properties.dimension = self.dimensions
        properties.leaf_capacity = self.max_entries
        properties.index_capacity = self.max_entries
        return rtree_index.Index(properties=properties)

    def _insert_record(self, record: Any, fallback_point: Any | None = None) -> None:
        point = self._record_point(record, fallback_point)
        record_id = self._next_id
        self._next_id += 1
        self._records[record_id] = record
        self._points[record_id] = point
        self._index.insert(record_id, self._point_mbr(point))

    def _range_search(self, min_corner: Iterable[float], max_corner: Iterable[float]) -> list[Any]:
        low = self._normalize_point(min_corner)
        high = self._normalize_point(max_corner)
        query_mbr = low + high
        record_ids = self._index.intersection(query_mbr)
        records = [
            self._records[record_id]
            for record_id in record_ids
            if self._inside_bounds(self._points[record_id], low, high)
        ]
        return sorted(records, key=self._record_point)

    def _knn_search(self, query: Any, k: int) -> list[Any]:
        point = self._normalize_point(query)
        nearest_ids = list(self._index.nearest(self._point_mbr(point), max(k, 0)))
        nearest_ids.sort(key=lambda record_id: self._distance(point, self._points[record_id]))
        return [self._records[record_id] for record_id in nearest_ids[:k]]

    def _point_search(self, point_like: Any) -> list[Any]:
        point = self._normalize_point(point_like)
        record_ids = self._index.intersection(self._point_mbr(point))
        return [
            self._records[record_id]
            for record_id in record_ids
            if self._points[record_id] == point
        ]

    def _record_point(self, record: Any, fallback_point: Any | None = None) -> tuple[float, ...]:
        if isinstance(record, dict) and self.column in record:
            return self._normalize_point(record[self.column])
        if not isinstance(record, dict) and hasattr(record, self.column):
            return self._normalize_point(getattr(record, self.column))
        if fallback_point is None:
            raise ValueError("R-Tree record does not contain the indexed point")
        return self._normalize_point(fallback_point)

    def _normalize_point(self, point: Any) -> tuple[float, ...]:
        if not isinstance(point, (list, tuple)):
            raise ValueError("R-Tree point must be a list or tuple")
        if len(point) != self.dimensions:
            raise ValueError("R-Tree point has invalid dimensions")
        return tuple(float(value) for value in point)

    def _point_mbr(self, point: tuple[float, ...]) -> tuple[float, ...]:
        return point + point

    def _inside_bounds(
        self,
        point: tuple[float, ...],
        low: tuple[float, ...],
        high: tuple[float, ...],
    ) -> bool:
        return all(low[index] <= point[index] <= high[index] for index in range(self.dimensions))

    def _distance(self, first: tuple[float, ...], second: tuple[float, ...]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))

    def _persist_snapshot(self) -> None:
        if self.buffer is None:
            return
        payload = {
            "column": self.column,
            "dimensions": self.dimensions,
            "max_entries": self.max_entries,
            "records": list(self._records.values()),
        }
        try:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            return
        self._write_page(0, encoded)
        self.buffer.flush(self.file_id)

    def _load_snapshot(self) -> None:
        if self.buffer is None:
            return
        raw = self._read_page(0)
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if payload.get("column") != self.column:
            return
        self._records = {}
        self._points = {}
        self._next_id = 1
        self._index = self._new_index()
        for record in payload.get("records", []):
            self._insert_record(record)

    def _read_page(self, page_no: int) -> bytes:
        return bytes(self.buffer.get(self.file_id, page_no).data)

    def _write_page(self, page_no: int, data: bytes) -> None:
        page = self.buffer.get(self.file_id, page_no)
        page.data[:] = data
        page.dirty = True

    def _stats(self) -> IOStats:
        if self.buffer is None:
            return IOStats()
        return self.buffer.stats()
