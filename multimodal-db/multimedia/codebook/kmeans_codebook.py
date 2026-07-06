from __future__ import annotations

import json
import pickle

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from core.ports.buffer import BufferManager
from core.ports.storage import StorageEngine
from multimedia.ports.codebook import Codebook

STATE_PAGE_SIZE = 4096


class KMeansCodebook(Codebook):
    # k es el número de grupos (visual words o acoustic words)
    def __init__(self, k: int = 256, random_state: int = 42, file_id: str = "codebook") -> None:
        self._k = k
        self._random_state = random_state
        self._kmeans = self._make_kmeans(k)
        self._fitted = False
        # Cada codebook guarda su estado bajo su propio nombre de archivo
        self.file_id = file_id
        # Peso IDF por cada visual word
        self._idf: np.ndarray = np.ones(k, dtype=np.float32)

    # Arma el modelo de clustering con la cantidad de grupos pedida
    def _make_kmeans(self, k: int) -> MiniBatchKMeans:
        return MiniBatchKMeans(
            n_clusters=k,
            random_state=self._random_state,
            batch_size=2048,
            n_init=3,
        )

    # Dice si el codebook ya aprendió sus centroides
    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, descriptors: np.ndarray) -> None:
        # No puede haber más grupos que descriptores disponibles
        k = min(self._k, int(descriptors.shape[0]))
        if k < 1:
            raise ValueError("no hay descriptores para entrenar el codebook")
        if k != self._k:
            self._k = k
            self._kmeans = self._make_kmeans(k)
            self._idf = np.ones(k, dtype=np.float32)
        # Aprende los centroides a partir de todos los descriptores
        self._kmeans.fit(descriptors)
        self._fitted = True

    def compute_idf(self, all_histograms: list[np.ndarray]) -> None:
        # Calcula cuántas imágenes contienen cada visual word
        n = len(all_histograms)
        df = np.zeros(self._k, dtype=np.float32)
        for hist in all_histograms:
            df += (hist > 0).astype(np.float32)
        # Evita división por cero con clip
        self._idf = np.log((n + 1) / (df + 1)).astype(np.float32)

    def quantize(self, descriptors: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Codebook no entrenado")
        if descriptors.shape[0] == 0:
            return np.zeros(self._k, dtype=np.float32)
        # Asigna cada descriptor al centroide más cercano
        labels = self._kmeans.predict(descriptors)
        # Genera el histograma de frecuencias normalizado
        histogram, _ = np.histogram(labels, bins=self._k, range=(0, self._k))
        total = histogram.sum()
        tf = (histogram / total).astype(np.float32) if total > 0 else histogram.astype(np.float32)
        # Pondera por IDF para reducir el peso de visual words comunes
        weighted = tf * self._idf
        norm = np.linalg.norm(weighted)
        if norm > 0:
            return (weighted / norm).astype(np.float32)
        return weighted

    def save(self, sink: StorageEngine) -> None:
        buffer = self._snapshot_buffer(sink)
        state = pickle.dumps({
            "k": self._k,
            "fitted": self._fitted,
            "kmeans": self._kmeans if self._fitted else None,
            "idf": self._idf,
        })
        # El estado se parte en páginas en lugar de un solo bloque grande
        pages = [
            state[start:start + STATE_PAGE_SIZE]
            for start in range(0, len(state), STATE_PAGE_SIZE)
        ]
        metadata = {"version": 2, "state_page_count": len(pages)}
        encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        self._write_page(buffer, 0, encoded)
        for page_no, page in enumerate(pages, start=1):
            self._write_page(buffer, page_no, page)
        buffer.flush(self.file_id)

    def load(self, source: StorageEngine) -> None:
        buffer = self._snapshot_buffer(source)
        raw = bytes(buffer.get(self.file_id, 0).data)
        if not raw:
            return
        try:
            metadata = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        stream = bytearray()
        for page_no in range(1, metadata.get("state_page_count", 0) + 1):
            stream.extend(buffer.get(self.file_id, page_no).data)
        if not stream:
            return
        state = pickle.loads(bytes(stream))
        self._k = state["k"]
        self._fitted = state["fitted"]
        self._idf = state["idf"]
        if state["kmeans"] is not None:
            self._kmeans = state["kmeans"]

    # Envuelve el storage en un buffer para respetar el camino único de I/O
    def _snapshot_buffer(self, storage: StorageEngine) -> BufferManager:
        from core.buffer.lru_buffer import LRUBufferManager

        return LRUBufferManager(storage)

    def _write_page(self, buffer: BufferManager, page_no: int, data: bytes) -> None:
        page = buffer.get(self.file_id, page_no)
        page.data[:] = data
        page.dirty = True
