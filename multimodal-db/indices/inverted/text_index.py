from __future__ import annotations

import json
from typing import Any, Iterable

from core.metrics import IOStats, OperationResult
from core.ports.index import Index
from core.ports.storage import StorageEngine
from indices.inverted.spimi_builder import SPIMIBlockBuilder, tokenize
from indices.ports import TextMatchPredicate


class InvertedIndex(Index):

    def __init__(
        self,
        column: str,
        block_document_limit: int = 128,
        storage: StorageEngine | None = None,
        file_id: str | None = None,
    ) -> None:
        if block_document_limit < 1:
            raise ValueError("SPIMI block document limit must be positive")
        self.column = column
        self.block_document_limit = block_document_limit
        self.storage = storage
        self.file_id = file_id or f"inverted_{column}"
        self._documents: dict[str, Any] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._last_block_count = 0
        self._load_snapshot()

    def build(self, records: Iterable[Any]) -> OperationResult:
        self._documents = {}
        documents: list[tuple[str, str]] = []
        for position, record in enumerate(records, start=1):
            doc_id = self._document_id(record, position)
            self._documents[doc_id] = record
            documents.append((doc_id, self._record_text(record)))
        builder = SPIMIBlockBuilder(block_document_limit=self.block_document_limit)
        self._postings = builder.build(documents)
        self._last_block_count = builder.block_count()
        self._persist_snapshot()
        return OperationResult(affected=len(self._documents), io=self._stats())

    def insert(self, key: Any, record: Any) -> OperationResult:
        doc_id = str(key)
        self._documents[doc_id] = record
        term_counts: dict[str, int] = {}
        for term in tokenize(self._record_text(record)):
            term_counts[term] = term_counts.get(term, 0) + 1
        for term, frequency in term_counts.items():
            postings = self._postings.setdefault(term, {})
            postings[doc_id] = postings.get(doc_id, 0) + frequency
        self._persist_snapshot()
        return OperationResult(affected=1, io=self._stats())

    def search(self, predicate: TextMatchPredicate | Any, k: int | None = None) -> OperationResult:
        terms = tokenize(predicate.terms if isinstance(predicate, TextMatchPredicate) else str(predicate))
        if not terms:
            return OperationResult(records=[], io=self._stats())
        matched_ids = set(self._postings.get(terms[0], {}))
        for term in terms[1:]:
            matched_ids &= set(self._postings.get(term, {}))
        records = [self._documents[doc_id] for doc_id in sorted(matched_ids) if doc_id in self._documents]
        limit = k if k is not None else getattr(predicate, "k", None)
        if limit is not None:
            records = records[:limit]
        return OperationResult(records=records, io=self._stats())

    def delete(self, key: Any) -> OperationResult:
        doc_id = str(key)
        if doc_id not in self._documents:
            return OperationResult(affected=0, io=self._stats())
        self._documents.pop(doc_id)
        empty_terms: list[str] = []
        for term, postings in self._postings.items():
            postings.pop(doc_id, None)
            if not postings:
                empty_terms.append(term)
        for term in empty_terms:
            self._postings.pop(term, None)
        self._persist_snapshot()
        return OperationResult(affected=1, io=self._stats())

    def postings_for(self, term: str) -> dict[str, int]:
        tokens = tokenize(term)
        if not tokens:
            return {}
        return dict(self._postings.get(tokens[0], {}))

    def block_count(self) -> int:
        return self._last_block_count

    def _document_id(self, record: Any, fallback: int) -> str:
        if isinstance(record, dict) and "id" in record:
            return str(record["id"])
        if not isinstance(record, dict) and hasattr(record, "id"):
            return str(getattr(record, "id"))
        return str(fallback)

    def _record_text(self, record: Any) -> str:
        if isinstance(record, dict):
            return str(record[self.column])
        return str(getattr(record, self.column))

    def _persist_snapshot(self) -> None:
        if self.storage is None:
            return
        payload = {
            "column": self.column,
            "documents": self._documents,
            "postings": self._postings,
            "last_block_count": self._last_block_count,
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
        self._documents = payload.get("documents", {})
        self._postings = payload.get("postings", {})
        self._last_block_count = payload.get("last_block_count", 0)

    def _stats(self) -> IOStats:
        if self.storage is None:
            return IOStats()
        return self.storage.stats()
