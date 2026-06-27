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

    def fit(self, descriptors: np.ndarray) -> None:
        # Aprende los centroides a partir de todos los descriptores
        self._kmeans.fit(descriptors)
        self._fitted = True

    def quantize(self, descriptors: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Codebook no entrenado")
        if descriptors.shape[0] == 0:
            return np.zeros(self._k, dtype=np.float32)
        # Asigna cada descriptor al centroide más cercano
        labels = self._kmeans.predict(descriptors)
        # Genera el histograma de frecuencias
        histogram, _ = np.histogram(labels, bins=self._k, range=(0, self._k))
        total = histogram.sum()
        tf = (histogram / total).astype(np.float32) if total > 0 else histogram.astype(np.float32)
        norm = np.linalg.norm(tf)
        if norm > 0:
            return (tf / norm).astype(np.float32)
        return tf

    def save(self, sink: StorageEngine) -> None:
        pass
