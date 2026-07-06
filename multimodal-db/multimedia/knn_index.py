from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from core.metrics import IOStats, OperationResult
from core.ports.buffer import BufferManager
from core.ports.index import Index, Key, Predicate, Record
from core.ports.storage import StorageEngine
from multimedia.ports.resolver import MediaResolver

SNAPSHOT_PAGE_SIZE = 4096


class MultimediaKNNIndex(Index):
    # Guarda histogramas en memoria indexados por clave
    def __init__(
        self,
        candidate_ratio: float = 0.01,
        resolver: MediaResolver | None = None,
        buffer: BufferManager | None = None,
        file_id: str | None = None,
    ) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        # Lista invertida: visual word -> lista de claves que la contienen
        self._inverted: dict[int, list[str]] = defaultdict(list)
        # Fracción mínima del índice que debe juntar la poda para usarse
        self._candidate_ratio = candidate_ratio
        # Convierte nombres de archivo en histogramas cuando está presente
        self._resolver = resolver
        # Sin buffer el índice trabaja solo en memoria
        self.buffer = buffer
        self.file_id = file_id or "knn_index"
        # Matriz con todos los vectores apilados para no rearmarla en cada búsqueda
        self._matrix: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._keys: list[str] = []
        self._positions: dict[str, int] = {}
        self._load_snapshot(self.buffer)

    def build(self, records: Iterable[Record]) -> OperationResult:
        # Cada record es una tupla (track_id, histograma)
        count = 0
        for key, vector in records:
            try:
                self._add_to_index(str(key), self._as_vector(vector))
            except (TypeError, ValueError) as error:
                return OperationResult.failure(str(error))
            count += 1
        self._persist_snapshot(self.buffer)
        return OperationResult(affected=count, io=self._stats())

    def insert(self, key: Key, record: Record) -> OperationResult:
        # Cuando llega la fila completa el vector sale de la clave
        value = key if isinstance(record, dict) else record
        # Un vector no identifica a la fila, la primera columna toma ese rol
        if isinstance(record, dict) and not isinstance(value, str):
            key = next(iter(record.values()), key)
        try:
            self._add_to_index(str(key), self._as_vector(value))
        except (TypeError, ValueError) as error:
            return OperationResult.failure(str(error))
        self._persist_snapshot(self.buffer)
        return OperationResult(affected=1, io=self._stats())

    def search(self, predicate: Predicate, k: int | None = None) -> OperationResult:
        if predicate is None:
            # Sin condición se devuelven todas las claves guardadas
            keys = list(self._vectors)
            if k is not None:
                keys = keys[:k]
            return OperationResult(records=[(key, 0.0) for key in keys])
        # Acepta un KnnPredicate, un histograma directo o un nombre de archivo
        if hasattr(predicate, "query"):
            if k is None:
                k = getattr(predicate, "k", None)
            predicate = predicate.query
        try:
            query = self._as_vector(predicate)
        except (TypeError, ValueError) as error:
            return OperationResult.failure(str(error))
        if len(self._vectors) == 0:
            return OperationResult(records=[])
        k = k or 10
        self._ensure_matrix()
        rows = self._candidate_rows(query, k)
        if rows is None:
            keys = self._keys
            matrix = self._matrix
            norms = self._norms
        else:
            keys = [self._keys[i] for i in rows]
            matrix = self._matrix[rows]
            norms = self._norms[rows]
        # Similitud coseno entre la consulta y los candidatos
        query_norm = np.linalg.norm(query)
        if query_norm == 0 or np.all(norms == 0):
            return OperationResult(records=[])
        similarities = (matrix @ query) / (norms * query_norm + 1e-9)
        top_k = int(min(k, len(keys)))
        indices = np.argpartition(similarities, -top_k)[-top_k:]
        indices = indices[np.argsort(similarities[indices])[::-1]]
        results = [(keys[i], float(similarities[i])) for i in indices]
        return OperationResult(records=results)

    def delete(self, key: Key) -> OperationResult:
        key = str(key)
        vector = self._vectors.pop(key, None)
        if vector is None:
            return OperationResult(affected=0)
        self._matrix = None
        # Elimina la clave de las listas invertidas donde aparece
        active_words = np.where(vector > 0)[0]
        for word in active_words:
            lst = self._inverted.get(int(word), [])
            if key in lst:
                lst.remove(key)
        self._persist_snapshot(self.buffer)
        return OperationResult(affected=1, io=self._stats())

    # Convierte el valor recibido en un vector usable
    def _as_vector(self, value: Any) -> np.ndarray:
        if isinstance(value, str):
            if self._resolver is None:
                raise ValueError(f"sin resolver configurado para el archivo: {value}")
            return np.asarray(self._resolver.resolve(value), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    # Rearma la matriz apilada solo cuando cambió el contenido del índice
    def _ensure_matrix(self) -> None:
        if self._matrix is not None:
            return
        self._keys = list(self._vectors)
        self._positions = {key: i for i, key in enumerate(self._keys)}
        self._matrix = np.stack([self._vectors[key] for key in self._keys])
        self._norms = np.linalg.norm(self._matrix, axis=1)

    def _add_to_index(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = vector
        self._matrix = None
        # Registra la clave en cada visual word activa del histograma
        active_words = np.where(vector > 0)[0]
        for word in active_words:
            self._inverted[int(word)].append(key)

    # Devuelve las filas candidatas para la búsqueda o None para revisar todo
    def _candidate_rows(self, query: np.ndarray, k: int) -> np.ndarray | None:
        total = len(self._keys)
        # Una word presente en muchos archivos no distingue a ninguno
        selective_cap = max(1, int(0.05 * total))
        # Solo las words raras de la consulta aportan a la poda
        selective_words = [
            int(word)
            for word in np.where(query > 0)[0]
            if 0 < len(self._inverted.get(int(word), [])) <= selective_cap
        ]
        if not selective_words:
            return None
        # La cantidad de candidatos protege el recall del top k pedido
        limit = max(k * 4, int(total * self._candidate_ratio))
        # Si la poda deja demasiados candidatos no vale la pena copiar el subset
        if limit >= 0.3 * total:
            return None
        # Puntaje de cada archivo según cuánto comparte las words raras de la consulta
        overlap = self._matrix[:, selective_words] @ query[selective_words]
        # Si pocos archivos comparten esas words se revisa todo el índice
        if int(np.count_nonzero(overlap)) < limit:
            return None
        return np.argpartition(overlap, -limit)[-limit:]

    def save(self, sink: StorageEngine) -> None:
        self._persist_snapshot(self._snapshot_buffer(sink))

    def load(self, source: StorageEngine) -> None:
        self._load_snapshot(self._snapshot_buffer(source))

    # Envuelve el storage en un buffer para respetar el camino único de I/O
    def _snapshot_buffer(self, storage: StorageEngine) -> BufferManager:
        from core.buffer.lru_buffer import LRUBufferManager

        return LRUBufferManager(storage)

    def _persist_snapshot(self, buffer: BufferManager | None) -> None:
        if buffer is None:
            return
        pages = self._encode_vector_pages()
        metadata = {"version": 1, "vector_page_count": len(pages)}
        encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        self._write_page(buffer, 0, encoded)
        for page_no, page in enumerate(pages, start=1):
            self._write_page(buffer, page_no, page)
        buffer.flush(self.file_id)

    # Cada fila del snapshot lleva una clave con su histograma
    # Las listas invertidas no se guardan porque se rearman al cargar
    def _encode_vector_pages(self) -> list[bytes]:
        rows = [
            json.dumps(
                {"key": key, "vector": vector.tolist()},
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            for key, vector in self._vectors.items()
        ]
        pages: list[bytes] = []
        current = bytearray()
        for row in rows:
            if current and len(current) + len(row) > SNAPSHOT_PAGE_SIZE:
                pages.append(bytes(current))
                current = bytearray()
            if len(row) > SNAPSHOT_PAGE_SIZE:
                pages.extend(self._split_large_row(row))
                continue
            current.extend(row)
        if current:
            pages.append(bytes(current))
        return pages

    def _split_large_row(self, row: bytes) -> list[bytes]:
        return [
            row[start:start + SNAPSHOT_PAGE_SIZE]
            for start in range(0, len(row), SNAPSHOT_PAGE_SIZE)
        ]

    def _load_snapshot(self, buffer: BufferManager | None) -> None:
        if buffer is None:
            return
        raw = bytes(buffer.get(self.file_id, 0).data)
        if not raw:
            return
        try:
            metadata = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        stream = bytearray()
        for page_no in range(1, metadata.get("vector_page_count", 0) + 1):
            stream.extend(buffer.get(self.file_id, page_no).data)
        self._vectors = {}
        self._inverted = defaultdict(list)
        self._matrix = None
        for line in stream.splitlines():
            if not line:
                continue
            row = json.loads(line.decode("utf-8"))
            self._add_to_index(row["key"], np.asarray(row["vector"], dtype=np.float32))

    def _write_page(self, buffer: BufferManager, page_no: int, data: bytes) -> None:
        page = buffer.get(self.file_id, page_no)
        page.data[:] = data
        page.dirty = True

    def _stats(self) -> IOStats:
        if self.buffer is None:
            return IOStats()
        return self.buffer.stats()
