from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.metrics import IOStats, OperationResult
from core.ports.index import Index
from core.ports.storage import StorageEngine
from indices.ports import EqualityPredicate, RangePredicate, SearchPredicate


@dataclass
class _LeafNode:
    keys: list[Any] = field(default_factory=list)
    values: list[list[Any]] = field(default_factory=list)
    next: "_LeafNode | None" = None


@dataclass
class _InternalNode:
    keys: list[Any] = field(default_factory=list)
    children: list["_Node"] = field(default_factory=list)


_Node = _LeafNode | _InternalNode


class BPlusTreeIndex(Index):

    def __init__(
        self,
        column: str,
        order: int = 4,
        storage: StorageEngine | None = None,
        file_id: str | None = None,
    ) -> None:
        if order < 3:
            raise ValueError("B+Tree order must be at least 3")
        self.column = column
        self.order = order
        self.max_keys = order - 1
        self.storage = storage
        self.file_id = file_id or f"bplus_{column}"
        self.root: _Node = _LeafNode()
        self._load_snapshot()

    def build(self, records: Iterable[Any]) -> OperationResult:
        self.root = _LeafNode()
        affected = 0
        for record in sorted(records, key=self._record_key):
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
        leaf = self._find_leaf(key)
        pos = bisect_left(leaf.keys, key)
        if pos >= len(leaf.keys) or leaf.keys[pos] != key:
            return OperationResult(affected=0, io=self._stats())
        removed = len(leaf.values[pos])
        leaf.keys.pop(pos)
        leaf.values.pop(pos)
        self._persist_snapshot()
        return OperationResult(affected=removed, io=self._stats())

    def _insert_record(self, key: Any, record: Any) -> None:
        path: list[tuple[_InternalNode, int]] = []
        leaf = self._find_leaf(key, path)
        pos = bisect_left(leaf.keys, key)
        if pos < len(leaf.keys) and leaf.keys[pos] == key:
            leaf.values[pos].append(record)
            return
        leaf.keys.insert(pos, key)
        leaf.values.insert(pos, [record])
        if len(leaf.keys) > self.max_keys:
            separator, right = self._split_leaf(leaf)
            self._insert_in_parent(path, separator, right)

    def _find_leaf(
        self,
        key: Any,
        path: list[tuple[_InternalNode, int]] | None = None,
    ) -> _LeafNode:
        node = self.root
        while isinstance(node, _InternalNode):
            child_pos = bisect_right(node.keys, key)
            if path is not None:
                path.append((node, child_pos))
            node = node.children[child_pos]
        return node

    def _split_leaf(self, leaf: _LeafNode) -> tuple[Any, _LeafNode]:
        mid = self._leaf_split_position(leaf.keys)
        right = _LeafNode(
            keys=leaf.keys[mid:],
            values=leaf.values[mid:],
            next=leaf.next,
        )
        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]
        leaf.next = right
        return right.keys[0], right

    def _leaf_split_position(self, keys: list[Any]) -> int:
        mid = len(keys) // 2
        while mid < len(keys) and keys[mid - 1] == keys[mid]:
            mid += 1
        if mid < len(keys):
            return mid
        mid = len(keys) // 2
        while mid > 1 and keys[mid - 1] == keys[mid]:
            mid -= 1
        return mid

    def _insert_in_parent(
        self,
        path: list[tuple[_InternalNode, int]],
        separator: Any,
        right_child: _Node,
    ) -> None:
        if not path:
            self.root = _InternalNode(keys=[separator], children=[self.root, right_child])
            return
        parent, child_pos = path.pop()
        parent.keys.insert(child_pos, separator)
        parent.children.insert(child_pos + 1, right_child)
        if len(parent.keys) > self.max_keys:
            promoted, right_internal = self._split_internal(parent)
            self._insert_in_parent(path, promoted, right_internal)

    def _split_internal(self, node: _InternalNode) -> tuple[Any, _InternalNode]:
        mid = len(node.keys) // 2
        promoted = node.keys[mid]
        right = _InternalNode(
            keys=node.keys[mid + 1:],
            children=node.children[mid + 1:],
        )
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        return promoted, right

    def _search_key(self, key: Any) -> list[Any]:
        leaf = self._find_leaf(key)
        pos = bisect_left(leaf.keys, key)
        if pos >= len(leaf.keys) or leaf.keys[pos] != key:
            return []
        return list(leaf.values[pos])

    def _search_range(self, predicate: RangePredicate) -> list[Any]:
        leaf = self._find_leaf(predicate.low)
        pos = bisect_left(leaf.keys, predicate.low)
        if not predicate.include_low:
            pos = bisect_right(leaf.keys, predicate.low)
        records: list[Any] = []
        while leaf is not None:
            while pos < len(leaf.keys):
                key = leaf.keys[pos]
                if key > predicate.high or (key == predicate.high and not predicate.include_high):
                    return records
                records.extend(leaf.values[pos])
                pos += 1
            leaf = leaf.next
            pos = 0
        return records

    def _record_key(self, record: Any) -> Any:
        if isinstance(record, dict):
            return record[self.column]
        return getattr(record, self.column)

    def _leftmost_leaf(self) -> _LeafNode:
        node = self.root
        while isinstance(node, _InternalNode):
            node = node.children[0]
        return node

    def _iter_entries(self) -> Iterable[tuple[Any, list[Any]]]:
        leaf: _LeafNode | None = self._leftmost_leaf()
        while leaf is not None:
            for key, values in zip(leaf.keys, leaf.values):
                yield key, list(values)
            leaf = leaf.next

    def _persist_snapshot(self) -> None:
        if self.storage is None:
            return
        payload = {
            "column": self.column,
            "entries": [
                {"key": key, "records": values}
                for key, values in self._iter_entries()
            ],
        }
        try:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            return
        self.storage.write_page(self.file_id, 0, encoded)

    def _load_snapshot(self) -> None:
        if self.storage is None:
            return
        raw = self.storage.read_page(self.file_id, 0)
        if not raw:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if payload.get("column") != self.column:
            return
        for entry in payload.get("entries", []):
            key = entry["key"]
            for record in entry.get("records", []):
                self._insert_record(key, record)

    def _stats(self) -> IOStats:
        if self.storage is None:
            return IOStats()
        return self.storage.stats()
