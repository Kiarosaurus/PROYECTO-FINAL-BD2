from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# Convierte un archivo en números
class FeatureExtractor(ABC):

    # Saca los números de un archivo
    @abstractmethod
    def extract(self, file_path: str) -> np.ndarray:
        ...

    # Tamaño de cada vector de salida
    @abstractmethod
    def feature_dim(self) -> int:
        ...

    # Formatos de archivo que acepta
    @abstractmethod
    def supported_formats(self) -> list[str]:
        ...
