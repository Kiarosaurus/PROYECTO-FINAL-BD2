from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from core.metrics import IOStats, OperationResult
from core.ports.storage import StorageEngine
from core.ports.buffer import BufferManager, Page
from core.ports.index import Index
from multimedia.ports.extractor import FeatureExtractor
from multimedia.ports.codebook import Codebook
from query.ports import Parser, Planner, Executor
from query.plan_types import PlanOp, QueryPlan, ResultSet
from query.index_factory import IndexFactory, IndexType


# Guarda las páginas en memoria
class MockStorageEngine(StorageEngine):

    def __init__(self) -> None:
        self._pages: dict[tuple[str, int], bytes] = {}
        self._next: dict[str, int] = {}
        self._stats = IOStats()

    def read_page(self, file_id: str, page_no: int) -> bytes:
        self._stats.add_read()
        return self._pages.get((file_id, page_no), b"")

    def write_page(self, file_id: str, page_no: int, data: bytes) -> None:
        self._stats.add_write()
        self._pages[(file_id, page_no)] = bytes(data)

    def allocate_page(self, file_id: str) -> int:
        page_no = self._next.get(file_id, 0)
        self._next[file_id] = page_no + 1
        self._pages[(file_id, page_no)] = b""
        self._stats.add_allocation()
        return page_no

    def stats(self) -> IOStats:
        return self._stats


# Cache simple encima de un storage
class MockBufferManager(BufferManager):

    def __init__(self, storage: StorageEngine) -> None:
        self._storage = storage
        self._cache: dict[tuple[str, int], Page] = {}

    def get(self, file_id: str, page_no: int) -> Page:
        key = (file_id, page_no)
        if key not in self._cache:
            data = bytearray(self._storage.read_page(file_id, page_no))
            self._cache[key] = Page(file_id, page_no, data)
        return self._cache[key]

    def pin(self, page: Page) -> None:
        page.pin_count += 1

    def flush(self, file_id: str | None = None) -> None:
        for (fid, page_no), page in list(self._cache.items()):
            if file_id is not None and fid != file_id:
                continue
            if page.dirty:
                self._storage.write_page(fid, page_no, bytes(page.data))
                page.dirty = False

    def stats(self) -> IOStats:
        return self._storage.stats()

    def allocate_page(self, file_id: str) -> int:
        return self._storage.allocate_page(file_id)


# Guarda las filas en una lista
class MockIndex(Index):

    def __init__(self) -> None:
        self._rows: list[Any] = []

    def build(self, records: Iterable[Any]) -> OperationResult:
        self._rows = list(records)
        return OperationResult(affected=len(self._rows))

    def insert(self, key: Any, record: Any) -> OperationResult:
        self._rows.append(record)
        return OperationResult(affected=1)

    def search(self, predicate: Any, k: int | None = None) -> OperationResult:
        rows = self._rows if k is None else self._rows[:k]
        return OperationResult(records=list(rows))

    def delete(self, key: Any) -> OperationResult:
        return OperationResult(affected=0)


# Devuelve siempre un vector de ceros
class MockFeatureExtractor(FeatureExtractor):

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def extract(self, file_path: str) -> np.ndarray:
        return np.zeros((1, self._dim), dtype=np.float32)

    def feature_dim(self) -> int:
        return self._dim

    def supported_formats(self) -> list[str]:
        return ["mock"]


# Manda todo al grupo cero
class MockCodebook(Codebook):

    def fit(self, descriptors: np.ndarray) -> None:
        return None

    def quantize(self, descriptors: np.ndarray) -> np.ndarray:
        return np.zeros(len(descriptors), dtype=np.int64)

    def save(self, sink: StorageEngine) -> None:
        return None


# Devuelve el mismo texto
class MockParser(Parser):

    def parse(self, sql: str) -> Any:
        return sql


# Arma un plan fijo
class MockPlanner(Planner):

    def plan(self, ast: Any, catalog: Any) -> QueryPlan:
        return QueryPlan(op=PlanOp.SELECT, table="mock")


# Devuelve un resultado vacío
class MockExecutor(Executor):

    def execute(self, plan: QueryPlan) -> ResultSet:
        return ResultSet()


# Entrega un MockIndex sin importar el tipo
class MockIndexFactory(IndexFactory):

    def __init__(self) -> None:
        self.created: list[IndexType] = []

    def create(self, index_type: IndexType, schema: Any, storage: Any) -> MockIndex:
        self.created.append(index_type)
        return MockIndex()
