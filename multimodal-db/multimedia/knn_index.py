from __future__ import annotations

import json
from collections import defaultdict
from typing import Iterable

import numpy as np

from core.metrics import IOStats, OperationResult
from core.ports.index import Index, Key, Predicate, Record
from core.ports.storage import StorageEngine


class MultimediaKNNIndex(Index):
    # Guarda histogramas en memoria indexados por clave
    def __init__(self, candidate_ratio: float = 0.5) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        # Lista invertida: visual word -> lista de claves que la contienen
        self._inverted: dict[int, list[str]] = defaultdict(list)
        # Fracción mínima de candidatos a revisar si el filtro es muy agresivo
        self._candidate_ratio = candidate_ratio

    def build(self, records: Iterable[Record]) -> OperationResult:
        # Cada record es una tupla (track_id, histograma)
        io = IOStats()
        count = 0
        for key, vector in records:
            self._add_to_index(str(key), np.asarray(vector, dtype=np.float32))
            count += 1
        return OperationResult(affected=count, io=io)

    def insert(self, key: Key, record: Record) -> OperationResult:
        self._add_to_index(str(key), np.asarray(record, dtype=np.float32))
        return OperationResult(affected=1)

    def search(self, predicate: Predicate, k: int | None = None) -> OperationResult:
        # Acepta un KnnPredicate o el histograma directo
        if hasattr(predicate, "query"):
            if k is None:
                k = getattr(predicate, "k", None)
            predicate = predicate.query
        query = np.asarray(predicate, dtype=np.float32)
        if len(self._vectors) == 0:
            return OperationResult(records=[])
        k = k or 10
        candidates = self._filter_candidates(query)
        if not candidates:
            candidates = list(self._vectors.keys())
        keys = list(candidates)
        matrix = np.stack([self._vectors[k_] for k_ in keys])
        # Similitud coseno entre la consulta y los candidatos
        norms = np.linalg.norm(matrix, axis=1)
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
        # Elimina la clave de las listas invertidas donde aparece
        active_words = np.where(vector > 0)[0]
        for word in active_words:
            lst = self._inverted.get(int(word), [])
            if key in lst:
                lst.remove(key)
        return OperationResult(affected=1)

    def _add_to_index(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = vector
        # Registra la clave en cada visual word activa del histograma
        active_words = np.where(vector > 0)[0]
        for word in active_words:
            self._inverted[int(word)].append(key)

    def _filter_candidates(self, query: np.ndarray) -> set[str]:
        # Busca las visual words activas en la consulta
        active_words = np.where(query > 0)[0]
        candidates: set[str] = set()
        for word in active_words:
            candidates.update(self._inverted.get(int(word), []))
        # Si hay muy pocos candidatos usa todo el índice
        min_candidates = max(1, int(len(self._vectors) * self._candidate_ratio))
        if len(candidates) < min_candidates:
            return set()
        return candidates

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
