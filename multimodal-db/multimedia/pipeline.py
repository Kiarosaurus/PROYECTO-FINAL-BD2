from __future__ import annotations

from pathlib import Path

import numpy as np

from core.metrics import OperationResult
from core.ports.index import Index
from core.ports.storage import StorageEngine
from multimedia.ports.codebook import Codebook
from multimedia.ports.extractor import FeatureExtractor


# Encadena extractor, codebook e índice KNN sobre archivos
class MultimediaPipeline:

    def __init__(self, extractor: FeatureExtractor, codebook: Codebook, index: Index) -> None:
        self._extractor = extractor
        self._codebook = codebook
        self._index = index

    # Construye el índice a partir de una lista de archivos
    def build_from_files(self, file_paths: list[str]) -> OperationResult:
        descriptors_by_key: dict[str, np.ndarray] = {}
        for file_path in file_paths:
            descriptors = self._extractor.extract(file_path)
            if descriptors.shape[0] == 0:
                continue
            descriptors_by_key[Path(file_path).name] = descriptors
        if not descriptors_by_key:
            return OperationResult.failure("ningún archivo produjo descriptores")
        stacked = np.vstack(list(descriptors_by_key.values()))
        self._codebook.fit(stacked)
        histograms = self._quantize_all(descriptors_by_key)
        # Con el IDF aprendido se recuantiza para aplicar los pesos nuevos
        if hasattr(self._codebook, "compute_idf"):
            self._codebook.compute_idf(list(histograms.values()))
            histograms = self._quantize_all(descriptors_by_key)
        return self._index.build(list(histograms.items()))

    # Busca los archivos más parecidos al archivo de consulta
    def search_file(self, file_path: str, k: int = 5) -> OperationResult:
        descriptors = self._extractor.extract(file_path)
        histogram = self._codebook.quantize(descriptors)
        return self._index.search(histogram, k=k)

    # Guarda codebook e índice usando el storage
    def save(self, sink: StorageEngine) -> None:
        self._codebook.save(sink)
        if hasattr(self._index, "save"):
            self._index.save(sink)

    def _quantize_all(self, descriptors_by_key: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            key: self._codebook.quantize(descriptors)
            for key, descriptors in descriptors_by_key.items()
        }
