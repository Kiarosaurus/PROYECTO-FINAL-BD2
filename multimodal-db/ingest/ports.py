from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from service.dto import ColumnSpec, IndexInfo


# Entrega un dataset como tabla, índices y filas listas para insertar
class DatasetLoader(ABC):

    # Nombre de la tabla destino
    @abstractmethod
    def table_name(self) -> str:
        ...

    # Columnas con su tipo
    @abstractmethod
    def columns(self) -> list[ColumnSpec]:
        ...

    # Índices que la tabla necesita
    @abstractmethod
    def indexes(self) -> list[IndexInfo]:
        ...

    # Filas en el mismo orden que las columnas declaradas
    @abstractmethod
    def rows(self) -> Iterator[tuple]:
        ...
