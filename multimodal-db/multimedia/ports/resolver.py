from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# Convierte el nombre de un archivo en su histograma
class MediaResolver(ABC):

    # Devuelve el histograma del archivo pedido
    @abstractmethod
    def resolve(self, file_path: str) -> np.ndarray:
        ...

    # Formatos de archivo que acepta
    @abstractmethod
    def supported_formats(self) -> list[str]:
        ...
