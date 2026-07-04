from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.metrics import IOStats, OperationResult
from core.ports.buffer import BufferManager
from core.ports.index import Index
from indices.ports import EqualityPredicate, RangePredicate, SearchPredicate


@dataclass
class _Bucket:
    local_depth: int
    entries: dict[str, list[Any]] = field(default_factory=dict)


class ExtendibleHashIndex(Index):

    def __init__(
        self,
        column: str,
        bucket_capacity: int = 4,
        buffer: BufferManager | None = None,
        file_id: str | None = None,
    ) -> None:
        if bucket_capacity < 1:
            raise ValueError("Hash bucket capacity must be positive")
        self.column = column
        self.bucket_capacity = bucket_capacity
        self.buffer = buffer
        self.file_id = file_id or f"hash_{column}"
        self.global_depth = 1
        self._buckets: list[_Bucket] = [_Bucket(local_depth=1), _Bucket(local_depth=1)]
        self._directory: list[int] = [0, 1]
        self._load_snapshot()

    def build(self, records: Iterable[Any]) -> OperationResult:
        self.global_depth = 1
        self._buckets = [_Bucket(local_depth=1), _Bucket(local_depth=1)]
        self._directory = [0, 1]
        affected = 0
        for record in records:
            self._insert_record(self._record_key(record), record)
            affected += 1
        self._persist_snapshot()
        return OperationResult(affected=affected, io=self._stats())

    def insert(self, key: Any, record: Any) -> OperationResult:
        self._insert_record(key, record)
        self._persist_snapshot()
        return OperationResult(affected=1, io=self._stats())

    def search(self, predicate: SearchPredicate | Any, k: int | None = None) -> OperationResult:
        if isinstance(predicate, EqualityPredicate):
            records = self._search_key(predicate.value)
        elif isinstance(predicate, RangePredicate):
            records = self._search_range(predicate)
        else:
            records = self._search_key(predicate)
        if k is not None:
            records = records[:k]
        return OperationResult(records=records, io=self._stats())

    def delete(self, key: Any) -> OperationResult:
        bucket = self._bucket_for_key(key)
        encoded_key = self._encode_key(key)
        records = bucket.entries.pop(encoded_key, [])
        if records:
            self._persist_snapshot()
        return OperationResult(affected=len(records), io=self._stats())

    def bucket_count(self) -> int:
        return len(self._buckets)

    def directory_size(self) -> int:
        return len(self._directory)

    def _insert_record(self, key: Any, record: Any) -> None:
        encoded_key = self._encode_key(key)
        while True:
            bucket = self._bucket_for_key(key)
            if encoded_key in bucket.entries:
                bucket.entries[encoded_key].append(record)
                return
            if len(bucket.entries) < self.bucket_capacity:
                bucket.entries[encoded_key] = [record]
                return
            self._split_bucket(key)

    def _split_bucket(self, key: Any) -> None:
        directory_pos = self._directory_position(key)
        bucket_index = self._directory[directory_pos]
        bucket = self._buckets[bucket_index]
        if bucket.local_depth == self.global_depth:
            self._directory.extend(self._directory)
            self.global_depth += 1
        old_depth = bucket.local_depth
        bucket.local_depth += 1
        new_bucket = _Bucket(local_depth=bucket.local_depth)
        new_bucket_index = len(self._buckets)
        self._buckets.append(new_bucket)
        if self.buffer is not None:
            self.buffer.allocate_page(f"{self.file_id}_bucket")
        split_bit = 1 << old_depth
        for pos, current_bucket_index in enumerate(self._directory):
            if current_bucket_index == bucket_index and pos & split_bit:
                self._directory[pos] = new_bucket_index
        old_entries = bucket.entries
        bucket.entries = {}
        for records in old_entries.values():
            for record in records:
                self._insert_record(self._record_key(record), record)

    def _search_key(self, key: Any) -> list[Any]:
        bucket = self._bucket_for_key(key)
        return list(bucket.entries.get(self._encode_key(key), []))

    def _search_range(self, predicate: RangePredicate) -> list[Any]:
        records: list[Any] = []
        for bucket in self._unique_buckets():
            for values in bucket.entries.values():
                for record in values:
                    key = self._record_key(record)
                    if key < predicate.low or (key == predicate.low and not predicate.include_low):
                        continue
                    if key > predicate.high or (key == predicate.high and not predicate.include_high):
                        continue
                    records.append(record)
        return sorted(records, key=self._record_key)

    def _unique_buckets(self) -> Iterable[_Bucket]:
        seen: set[int] = set()
        for bucket_index in self._directory:
            if bucket_index in seen:
                continue
            seen.add(bucket_index)
            yield self._buckets[bucket_index]

    def _bucket_for_key(self, key: Any) -> _Bucket:
        return self._buckets[self._directory[self._directory_position(key)]]

    def _directory_position(self, key: Any) -> int:
        return self._hash_key(key) & ((1 << self.global_depth) - 1)

    def _hash_key(self, key: Any) -> int:
        encoded = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.blake2b(encoded, digest_size=8).digest()
        return int.from_bytes(digest, "little")

    def _encode_key(self, key: Any) -> str:
        return json.dumps(key, sort_keys=True, separators=(",", ":"))

    def _record_key(self, record: Any) -> Any:
        if isinstance(record, dict):
            return record[self.column]
        return getattr(record, self.column)

    def _persist_snapshot(self) -> None:
        if self.buffer is None:
            return
        payload = {
            "column": self.column,
            "bucket_capacity": self.bucket_capacity,
            "global_depth": self.global_depth,
            "directory": self._directory,
            "buckets": [
                {
                    "local_depth": bucket.local_depth,
                    "entries": bucket.entries,
                }
                for bucket in self._buckets
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
        self.global_depth = payload.get("global_depth", 1)
        self._directory = list(payload.get("directory", [0, 1]))
        self._buckets = [
            _Bucket(
                local_depth=bucket["local_depth"],
                entries=bucket.get("entries", {}),
            )
            for bucket in payload.get("buckets", [])
        ]
        if not self._buckets:
            self._buckets = [_Bucket(local_depth=1), _Bucket(local_depth=1)]
            self._directory = [0, 1]
            self.global_depth = 1

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
