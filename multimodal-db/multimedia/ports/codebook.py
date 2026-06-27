from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.ports.storage import StorageEngine


# Agrupa los números en grupos parecidos
class Codebook(ABC):

    # Aprende los grupos a partir de los datos
    @abstractmethod
    def fit(self, descriptors: np.ndarray) -> None:
        ...

    # Asigna cada dato a su grupo
    @abstractmethod
    def quantize(self, descriptors: np.ndarray) -> np.ndarray:
        ...

    # Guarda el codebook usando el storage
    @abstractmethod
    def save(self, sink: StorageEngine) -> None:
        ...
