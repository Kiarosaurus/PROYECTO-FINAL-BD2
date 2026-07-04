from __future__ import annotations

import json
import math
from typing import Any, Iterable

from core.metrics import IOStats, OperationResult
from core.ports.buffer import BufferManager
from core.ports.index import Index
from indices.inverted.spimi_builder import SPIMIBlockBuilder, tokenize
from indices.inverted.text_preprocessor import DEFAULT_PREPROCESSOR, TextPreprocessor
from indices.ports import TextMatchPredicate

POSTING_PAGE_SIZE = 4096


class InvertedIndex(Index):

    def __init__(
        self,
        column: str,
        block_document_limit: int = 128,
        buffer: BufferManager | None = None,
        file_id: str | None = None,
        preprocessor: TextPreprocessor = DEFAULT_PREPROCESSOR,
    ) -> None:
        if block_document_limit < 1:
            raise ValueError("SPIMI block document limit must be positive")
        self.column = column
        self.block_document_limit = block_document_limit
        self.buffer = buffer
        self.file_id = file_id or f"inverted_{column}"
        self.preprocessor = preprocessor
        self._documents: dict[str, Any] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_norms: dict[str, float] = {}
        self._last_block_count = 0
        self._posting_page_count = 0
        self._load_snapshot()

    def build(self, records: Iterable[Any]) -> OperationResult:
        self._documents = {}
        documents: list[tuple[str, str]] = []
        for position, record in enumerate(records, start=1):
            doc_id = self._document_id(record, position)
            self._documents[doc_id] = record
            documents.append((doc_id, self._record_text(record)))
        builder = SPIMIBlockBuilder(
            block_document_limit=self.block_document_limit,
            preprocessor=self.preprocessor,
        )
        self._postings = builder.build(documents)
        self._last_block_count = builder.block_count()
        self._compute_document_norms()
        self._persist_snapshot()
        return OperationResult(affected=len(self._documents), io=self._stats())

    def insert(self, key: Any, record: Any) -> OperationResult:
        doc_id = str(key)
        self._documents[doc_id] = record
        term_counts: dict[str, int] = {}
        for term in self.preprocessor.tokenize(self._record_text(record)):
            term_counts[term] = term_counts.get(term, 0) + 1
        for term, frequency in term_counts.items():
            postings = self._postings.setdefault(term, {})
            postings[doc_id] = postings.get(doc_id, 0) + frequency
        self._compute_document_norms()
        self._persist_snapshot()
        return OperationResult(affected=1, io=self._stats())

    def search(self, predicate: TextMatchPredicate | Any, k: int | None = None) -> OperationResult:
        terms = self.preprocessor.tokenize(
            predicate.terms if isinstance(predicate, TextMatchPredicate) else str(predicate)
        )
        if not terms:
            return OperationResult(records=[], io=self._stats())
        limit = k if k is not None else getattr(predicate, "k", None)
        ranked = self.rank(predicate.terms if isinstance(predicate, TextMatchPredicate) else str(predicate), limit)
        records = [self._documents[doc_id] for doc_id, _score in ranked if doc_id in self._documents]
        return OperationResult(records=records, io=self._stats())

    def rank(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        query_tf: dict[str, int] = {}
        for term in self.preprocessor.tokenize(query):
            query_tf[term] = query_tf.get(term, 0) + 1
        if not query_tf:
            return []
        query_weights: dict[str, float] = {}
        for term, frequency in query_tf.items():
            if term not in self._postings:
                continue
            query_weights[term] = frequency * self._idf(term)
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if not query_weights or query_norm == 0.0:
            return []
        scores: dict[str, float] = {}
        for term, query_weight in query_weights.items():
            idf = self._idf(term)
            for doc_id, frequency in self._postings.get(term, {}).items():
                doc_weight = frequency * idf
                scores[doc_id] = scores.get(doc_id, 0.0) + doc_weight * query_weight
        ranked: list[tuple[str, float]] = []
        for doc_id, dot_product in scores.items():
            doc_norm = self._doc_norms.get(doc_id, 0.0)
            if doc_norm == 0.0:
                continue
            ranked.append((doc_id, dot_product / (doc_norm * query_norm)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        if k is not None:
            return ranked[:k]
        return ranked

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
        self._compute_document_norms()
        self._persist_snapshot()
        return OperationResult(affected=1, io=self._stats())

    def postings_for(self, term: str) -> dict[str, int]:
        tokens = self.preprocessor.tokenize(term)
        if not tokens:
            return {}
        return dict(self._postings.get(tokens[0], {}))

    def block_count(self) -> int:
        return self._last_block_count

    def posting_page_count(self) -> int:
        return self._posting_page_count

    def document_norm(self, doc_id: Any) -> float:
        return self._doc_norms.get(str(doc_id), 0.0)

    def _compute_document_norms(self) -> None:
        norm_squares: dict[str, float] = {doc_id: 0.0 for doc_id in self._documents}
        for term, postings in self._postings.items():
            idf = self._idf(term)
            for doc_id, frequency in postings.items():
                weight = frequency * idf
                norm_squares[doc_id] = norm_squares.get(doc_id, 0.0) + weight * weight
        self._doc_norms = {
            doc_id: math.sqrt(value)
            for doc_id, value in norm_squares.items()
        }

    def _idf(self, term: str) -> float:
        total_docs = max(len(self._documents), 1)
        document_frequency = len(self._postings.get(term, {}))
        return math.log((total_docs + 1) / (document_frequency + 1)) + 1.0

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
        if self.buffer is None:
            return
        posting_pages = self._encode_posting_pages()
        metadata = {
            "version": 2,
            "column": self.column,
            "documents": self._documents,
            "doc_norms": self._doc_norms,
            "last_block_count": self._last_block_count,
            "posting_page_count": len(posting_pages),
        }
        try:
            encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            return
        self._write_page(0, encoded)
        for page_no, page in enumerate(posting_pages, start=1):
            self._write_page(page_no, page)
        self.buffer.flush(self.file_id)
        self._posting_page_count = len(posting_pages)

    def _encode_posting_pages(self) -> list[bytes]:
        rows = [
            json.dumps(
                {"term": term, "postings": postings},
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            for term, postings in sorted(self._postings.items())
        ]
        pages: list[bytes] = []
        current = bytearray()
        for row in rows:
            if current and len(current) + len(row) > POSTING_PAGE_SIZE:
                pages.append(bytes(current))
                current = bytearray()
            if len(row) > POSTING_PAGE_SIZE:
                pages.extend(self._split_large_row(row))
                continue
            current.extend(row)
        if current:
            pages.append(bytes(current))
        return pages

    def _split_large_row(self, row: bytes) -> list[bytes]:
        return [
            row[start:start + POSTING_PAGE_SIZE]
            for start in range(0, len(row), POSTING_PAGE_SIZE)
        ]

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
        self._documents = payload.get("documents", {})
        self._doc_norms = payload.get("doc_norms", {})
        self._last_block_count = payload.get("last_block_count", 0)
        self._posting_page_count = payload.get("posting_page_count", 0)
        if "postings" in payload:
            self._postings = payload.get("postings", {})
        else:
            self._postings = self._read_posting_pages(self._posting_page_count)
        if not self._doc_norms:
            self._compute_document_norms()

    def _read_posting_pages(self, page_count: int) -> dict[str, dict[str, int]]:
        raw = bytearray()
        for page_no in range(1, page_count + 1):
            raw.extend(self._read_page(page_no))
        postings: dict[str, dict[str, int]] = {}
        for line in raw.splitlines():
            if not line:
                continue
            item = json.loads(line.decode("utf-8"))
            postings[item["term"]] = item["postings"]
        return postings

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
