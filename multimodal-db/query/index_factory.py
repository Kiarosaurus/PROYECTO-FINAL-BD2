from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from core.ports.index import Index
from core.ports.storage import StorageEngine

# Resuelto por la capa de servicio cuando exista
Schema = Any


# Nombres de los índices que se pueden crear
class IndexType(Enum):
    BPLUS = "bplus"
    ISAM = "isam"
    HASH = "hash"
    RTREE = "rtree"
    INVERTED = "inverted"
    KNN = "knn"

    # Busca el tipo a partir de su nombre en texto
    @classmethod
    def from_name(cls, name: str) -> "IndexType":
        return cls(name.lower())


# Crea el índice del tipo pedido
class IndexFactory(ABC):

    @abstractmethod
    def create(
        self,
        index_type: IndexType,
        schema: Schema,
        storage: StorageEngine,
    ) -> Index:
        ...
