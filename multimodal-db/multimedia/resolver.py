from __future__ import annotations

from pathlib import Path

import numpy as np

from core.ports.storage import StorageEngine
from multimedia.ports.codebook import Codebook
from multimedia.ports.extractor import FeatureExtractor
from multimedia.ports.resolver import MediaResolver


# Resuelve archivos con un extractor y un codebook
class PipelineMediaResolver(MediaResolver):

    # media_dir es la carpeta donde viven los archivos subidos
    # Con storage presente el codebook se recupera y se guarda ahí
    def __init__(
        self,
        extractor: FeatureExtractor,
        codebook: Codebook,
        media_dir: str | Path,
        storage: StorageEngine | None = None,
    ) -> None:
        self._extractor = extractor
        self._codebook = codebook
        self._media_dir = Path(media_dir)
        self._storage = storage
        self._fitted = False
        self._restore_codebook()

    # Intenta recuperar un codebook ya entrenado para no repetir el fit
    def _restore_codebook(self) -> None:
        if self._storage is None:
            return
        loader = getattr(self._codebook, "load", None)
        if callable(loader):
            loader(self._storage)
        if getattr(self._codebook, "is_fitted", False):
            self._fitted = True

    def resolve(self, file_path: str) -> np.ndarray:
        path = self._locate(file_path)
        self._ensure_fitted()
        descriptors = self._extractor.extract(str(path))
        return self._codebook.quantize(descriptors)

    def supported_formats(self) -> list[str]:
        return self._extractor.supported_formats()

    # Busca el archivo como ruta directa, en la carpeta o en sus subcarpetas
    def _locate(self, file_path: str) -> Path:
        direct = Path(file_path)
        if direct.is_file():
            return direct
        candidate = self._media_dir / direct.name
        if candidate.is_file():
            return candidate
        if self._media_dir.is_dir():
            for match in sorted(self._media_dir.rglob(direct.name)):
                if match.is_file():
                    return match
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
        # El codebook recién entrenado queda guardado para el próximo arranque
        if self._storage is not None:
            self._codebook.save(self._storage)

    # Junta los descriptores de todos los archivos soportados
    def _corpus_descriptors(self) -> dict[str, np.ndarray]:
        formats = self._extractor.supported_formats()
        corpus: dict[str, np.ndarray] = {}
        if not self._media_dir.is_dir():
            return corpus
        for path in sorted(self._media_dir.rglob("*")):
            if not path.is_file():
                continue
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


# Enruta cada archivo al resolver que acepta su extensión
class CompositeMediaResolver(MediaResolver):

    def __init__(self, resolvers: list[MediaResolver]) -> None:
        self._resolvers = list(resolvers)

    def resolve(self, file_path: str) -> np.ndarray:
        suffix = Path(file_path).suffix.lower()
        for resolver in self._resolvers:
            if suffix in resolver.supported_formats():
                return resolver.resolve(file_path)
        raise ValueError(f"formato no soportado: {file_path}")

    def supported_formats(self) -> list[str]:
        formats: list[str] = []
        for resolver in self._resolvers:
            for fmt in resolver.supported_formats():
                if fmt not in formats:
                    formats.append(fmt)
        return formats
