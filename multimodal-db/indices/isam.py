from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.metrics import IOStats, OperationResult
from core.ports.buffer import BufferManager
from core.ports.index import Index
from indices.ports import EqualityPredicate, RangePredicate, SearchPredicate


@dataclass
class _OverflowPage:
    records: list[Any] = field(default_factory=list)
    next: int | None = None


@dataclass
class _PrimaryPage:
    high_key: Any
    records: list[Any] = field(default_factory=list)
    overflow_head: int | None = None


class ISAMIndex(Index):

    def __init__(
        self,
        column: str,
        page_capacity: int = 4,
        overflow_capacity: int = 3,
        buffer: BufferManager | None = None,
        file_id: str | None = None,
    ) -> None:
        if page_capacity < 1:
            raise ValueError("ISAM page capacity must be positive")
        if overflow_capacity < 1:
            raise ValueError("ISAM overflow capacity must be positive")
        self.column = column
        self.page_capacity = page_capacity
        self.overflow_capacity = overflow_capacity
        self.buffer = buffer
        self.file_id = file_id or f"isam_{column}"
        self._primary_pages: list[_PrimaryPage] = []
        self._overflow_pages: list[_OverflowPage] = []
        self._load_snapshot()

    def build(self, records: Iterable[Any]) -> OperationResult:
        sorted_records = sorted(records, key=self._record_key)
        self._primary_pages = []
        self._overflow_pages = []
        for start in range(0, len(sorted_records), self.page_capacity):
            page_records = sorted_records[start:start + self.page_capacity]
            high_key = self._record_key(page_records[-1])
            self._primary_pages.append(_PrimaryPage(high_key=high_key, records=page_records))
        self._persist_snapshot()
        return OperationResult(affected=len(sorted_records), io=self._stats())

    def insert(self, key: Any, record: Any) -> OperationResult:
        if not self._primary_pages:
            self._primary_pages.append(_PrimaryPage(high_key=key, records=[record]))
            self._persist_snapshot()
            return OperationResult(affected=1, io=self._stats())
        page = self._find_page(key)
        if key > page.high_key:
            page.high_key = key
        if len(page.records) < self.page_capacity:
            self._insert_sorted(page.records, record)
        else:
            self._append_overflow_record(page, record)
        self._persist_snapshot()
        return OperationResult(affected=1, io=self._stats())

    def search(self, predicate: SearchPredicate | Any, k: int | None = None) -> OperationResult:
        if predicate is None:
            # Sin condición se recorren todas las páginas en orden
            records = [
                record
                for page in self._primary_pages
                for record in self._page_records(page)
            ]
        elif isinstance(predicate, EqualityPredicate):
            records = self._search_key(predicate.value)
        elif isinstance(predicate, RangePredicate):
            records = self._search_range(predicate)
        else:
            records = self._search_key(predicate)
        if k is not None:
            records = records[:k]
        return OperationResult(records=records, io=self._stats())

    def delete(self, key: Any) -> OperationResult:
        if not self._primary_pages:
            return OperationResult(affected=0, io=self._stats())
        affected = 0
        for page in self._candidate_pages(key):
            kept = [record for record in page.records if self._record_key(record) != key]
            affected += len(page.records) - len(kept)
            page.records = kept
            current = page.overflow_head
            while current is not None:
                overflow = self._overflow_pages[current]
                kept_overflow = [
                    record
                    for record in overflow.records
                    if self._record_key(record) != key
                ]
                affected += len(overflow.records) - len(kept_overflow)
                overflow.records = kept_overflow
                current = overflow.next
        if affected:
            self._persist_snapshot()
        return OperationResult(affected=affected, io=self._stats())

    def overflow_page_count(self) -> int:
        return len(self._overflow_pages)

    def _find_page(self, key: Any) -> _PrimaryPage:
        if not self._primary_pages:
            raise ValueError("ISAM index is empty")
        fences = [page.high_key for page in self._primary_pages]
        page_pos = bisect_left(fences, key)
        if page_pos >= len(self._primary_pages):
            page_pos = len(self._primary_pages) - 1
        return self._primary_pages[page_pos]

    def _candidate_pages(self, key: Any) -> list[_PrimaryPage]:
        if not self._primary_pages:
            return []
        page = self._find_page(key)
        pages = [page]
        page_pos = self._primary_pages.index(page)
        # Si la última clave de una página es la buscada, los duplicados pueden seguir en la próxima
        while page_pos + 1 < len(self._primary_pages) and self._primary_pages[page_pos].high_key == key:
            page_pos += 1
            pages.append(self._primary_pages[page_pos])
        return pages

    def _append_overflow_record(self, page: _PrimaryPage, record: Any) -> None:
        if page.overflow_head is None:
            page.overflow_head = self._new_overflow_page(record)
            return
        current = page.overflow_head
        while current is not None:
            overflow = self._overflow_pages[current]
            if len(overflow.records) < self.overflow_capacity:
                self._insert_sorted(overflow.records, record)
                return
            if overflow.next is None:
                overflow.next = self._new_overflow_page(record)
                return
            current = overflow.next

    def _new_overflow_page(self, record: Any) -> int:
        page_no = len(self._overflow_pages)
        self._overflow_pages.append(_OverflowPage(records=[record]))
        if self.buffer is not None:
            self.buffer.allocate_page(f"{self.file_id}_overflow")
        return page_no

    def _search_key(self, key: Any) -> list[Any]:
        records: list[Any] = []
        for page in self._candidate_pages(key):
            for record in self._page_records(page):
                if self._record_key(record) == key:
                    records.append(record)
        return records

    def _search_range(self, predicate: RangePredicate) -> list[Any]:
        records: list[Any] = []
        for page in self._primary_pages:
            if page.high_key < predicate.low:
                continue
            for record in self._page_records(page):
                key = self._record_key(record)
                if key < predicate.low or (key == predicate.low and not predicate.include_low):
                    continue
                if key > predicate.high or (key == predicate.high and not predicate.include_high):
                    continue
                records.append(record)
        return sorted(records, key=self._record_key)

    def _page_records(self, page: _PrimaryPage) -> Iterable[Any]:
        yield from page.records
        current = page.overflow_head
        while current is not None:
            overflow = self._overflow_pages[current]
            yield from overflow.records
            current = overflow.next

    def _insert_sorted(self, records: list[Any], record: Any) -> None:
        key = self._record_key(record)
        keys = [self._record_key(current) for current in records]
        records.insert(bisect_right(keys, key), record)

    def _record_key(self, record: Any) -> Any:
        if isinstance(record, dict):
            return record[self.column]
        return getattr(record, self.column)

    def _persist_snapshot(self) -> None:
        if self.buffer is None:
            return
        payload = {
            "column": self.column,
            "page_capacity": self.page_capacity,
            "overflow_capacity": self.overflow_capacity,
            "primary_pages": [
                {
                    "high_key": page.high_key,
                    "records": page.records,
                    "overflow_head": page.overflow_head,
                }
                for page in self._primary_pages
            ],
            "overflow_pages": [
                {"records": page.records, "next": page.next}
                for page in self._overflow_pages
            ],
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
        self._primary_pages = [
            _PrimaryPage(
                high_key=page["high_key"],
                records=page.get("records", []),
                overflow_head=page.get("overflow_head"),
            )
            for page in payload.get("primary_pages", [])
        ]
        self._overflow_pages = [
            _OverflowPage(
                records=page.get("records", []),
                next=page.get("next"),
            )
            for page in payload.get("overflow_pages", [])
        ]

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
