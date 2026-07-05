from __future__ import annotations

from pathlib import Path

import numpy as np

from multimedia.ports.codebook import Codebook
from multimedia.ports.extractor import FeatureExtractor
from multimedia.ports.resolver import MediaResolver


# Resuelve archivos con un extractor y un codebook
class PipelineMediaResolver(MediaResolver):

    # media_dir es la carpeta donde viven los archivos subidos
    def __init__(
        self,
        extractor: FeatureExtractor,
        codebook: Codebook,
        media_dir: str | Path,
    ) -> None:
        self._extractor = extractor
        self._codebook = codebook
        self._media_dir = Path(media_dir)
        self._fitted = False

    def resolve(self, file_path: str) -> np.ndarray:
        path = self._locate(file_path)
        self._ensure_fitted()
        descriptors = self._extractor.extract(str(path))
        return self._codebook.quantize(descriptors)

    def supported_formats(self) -> list[str]:
        return self._extractor.supported_formats()

    # Busca el archivo como ruta directa o dentro de la carpeta de medios
    def _locate(self, file_path: str) -> Path:
        direct = Path(file_path)
        if direct.is_file():
            return direct
        candidate = self._media_dir / direct.name
        if candidate.is_file():
            return candidate
        raise ValueError(f"archivo no encontrado: {file_path}")

    # Entrena el codebook una sola vez con los archivos de la carpeta
    def _ensure_fitted(self) -> None:
        if self._fitted:
            return
        corpus = self._corpus_descriptors()
        if not corpus:
            raise ValueError("no hay archivos para entrenar el codebook")
        self._codebook.fit(np.vstack(list(corpus.values())))
        # Con el IDF aprendido los pesos de cada grupo quedan mejor repartidos
        if hasattr(self._codebook, "compute_idf"):
            histograms = [self._codebook.quantize(d) for d in corpus.values()]
            self._codebook.compute_idf(histograms)
        self._fitted = True

    # Junta los descriptores de todos los archivos soportados
    def _corpus_descriptors(self) -> dict[str, np.ndarray]:
        formats = self._extractor.supported_formats()
        corpus: dict[str, np.ndarray] = {}
        if not self._media_dir.is_dir():
            return corpus
        for path in sorted(self._media_dir.iterdir()):
            if path.suffix.lower() not in formats:
                continue
            try:
                descriptors = self._extractor.extract(str(path))
            except ValueError:
                continue
            if descriptors.shape[0] == 0:
                continue
            corpus[path.name] = descriptors
        return corpus
