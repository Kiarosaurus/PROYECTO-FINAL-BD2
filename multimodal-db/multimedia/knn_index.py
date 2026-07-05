from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from core.metrics import IOStats, OperationResult
from core.ports.index import Index, Key, Predicate, Record
from core.ports.storage import StorageEngine
from multimedia.ports.resolver import MediaResolver


class MultimediaKNNIndex(Index):
    # Guarda histogramas en memoria indexados por clave
    def __init__(
        self,
        candidate_ratio: float = 0.01,
        resolver: MediaResolver | None = None,
    ) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        # Lista invertida: visual word -> lista de claves que la contienen
        self._inverted: dict[int, list[str]] = defaultdict(list)
        # Fracción mínima del índice que debe juntar la poda para usarse
        self._candidate_ratio = candidate_ratio
        # Convierte nombres de archivo en histogramas cuando está presente
        self._resolver = resolver
        # Matriz con todos los vectores apilados para no rearmarla en cada búsqueda
        self._matrix: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._keys: list[str] = []
        self._positions: dict[str, int] = {}

    def build(self, records: Iterable[Record]) -> OperationResult:
        # Cada record es una tupla (track_id, histograma)
        io = IOStats()
        count = 0
        for key, vector in records:
            try:
                self._add_to_index(str(key), self._as_vector(vector))
            except (TypeError, ValueError) as error:
                return OperationResult.failure(str(error))
            count += 1
        return OperationResult(affected=count, io=io)

    def insert(self, key: Key, record: Record) -> OperationResult:
        # Cuando llega la fila completa el vector sale de la clave
        value = key if isinstance(record, dict) else record
        try:
            self._add_to_index(str(key), self._as_vector(value))
        except (TypeError, ValueError) as error:
            return OperationResult.failure(str(error))
        return OperationResult(affected=1)

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
        return OperationResult(affected=1)

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
        # Serializa histogramas y lista invertida como JSON en una página
        state = {
            "vectors": {key: vector.tolist() for key, vector in self._vectors.items()},
            "inverted": {str(word): keys for word, keys in self._inverted.items()},
        }
        data = json.dumps(state, separators=(",", ":")).encode("utf-8")
        sink.write_page("knn_index", 0, data)

    def load(self, source: StorageEngine) -> None:
        # Trae de vuelta histogramas y lista invertida guardados antes
        data = source.read_page("knn_index", 0)
        if not data:
            return
        state = json.loads(data.decode("utf-8"))
        self._vectors = {
            key: np.asarray(vector, dtype=np.float32)
            for key, vector in state["vectors"].items()
        }
        self._inverted = defaultdict(
            list,
            {int(word): list(keys) for word, keys in state["inverted"].items()},
        )
        self._matrix = None
