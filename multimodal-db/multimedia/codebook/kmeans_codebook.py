from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from core.ports.storage import StorageEngine
from multimedia.ports.codebook import Codebook


class KMeansCodebook(Codebook):
    # k es el número de grupos (visual words o acoustic words)
    def __init__(self, k: int = 256, random_state: int = 42) -> None:
        self._k = k
        self._kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=2048,
            n_init=3,
        )
        self._fitted = False
        # Peso IDF por cada visual word
        self._idf: np.ndarray = np.ones(k, dtype=np.float32)

    def fit(self, descriptors: np.ndarray) -> None:
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
        # Serializa centroides y estado en una sola página
        import pickle
        data = pickle.dumps({
            "k": self._k,
            "fitted": self._fitted,
            "kmeans": self._kmeans if self._fitted else None,
            "idf": self._idf,
        })
        sink.write_page("codebook", 0, data)

    def load(self, source: StorageEngine) -> None:
        # Trae de vuelta centroides y estado guardados antes
        import pickle
        data = source.read_page("codebook", 0)
        if not data:
            return
        state = pickle.loads(data)
        self._k = state["k"]
        self._fitted = state["fitted"]
        self._idf = state["idf"]
        if state["kmeans"] is not None:
            self._kmeans = state["kmeans"]
