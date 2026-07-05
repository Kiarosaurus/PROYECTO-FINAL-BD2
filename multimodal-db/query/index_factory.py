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


# Fábrica que entrega los índices reales del engine
class EngineIndexFactory(IndexFactory):

    def __init__(self, buffer_capacity: int = 64, media_resolver: Any = None) -> None:
        self._buffer_capacity = buffer_capacity
        # Un solo buffer por storage para que el I/O se cuente en un solo lugar
        self._buffers: dict[int, Any] = {}
        # Resolver de archivos compartido por los índices multimedia
        self._media_resolver = media_resolver

    def create(
        self,
        index_type: IndexType,
        schema: Schema,
        storage: StorageEngine,
    ) -> Index:
        from indices.bplus_tree import BPlusTreeIndex
        from indices.extendible_hash import ExtendibleHashIndex
        from indices.inverted.text_index import InvertedIndex
        from indices.isam import ISAMIndex
        from indices.rtree import RTreeIndex
        from multimedia.knn_index import MultimediaKNNIndex

        if index_type is IndexType.KNN:
            return MultimediaKNNIndex(resolver=self._media_resolver)
        table, column = self._table_and_column(schema)
        buffer = self._buffer_for(storage)
        file_id = f"{index_type.value}_{table}_{column}"
        if index_type is IndexType.BPLUS:
            return BPlusTreeIndex(column=column, buffer=buffer, file_id=file_id)
        if index_type is IndexType.ISAM:
            return ISAMIndex(column=column, buffer=buffer, file_id=file_id)
        if index_type is IndexType.HASH:
            return ExtendibleHashIndex(column=column, buffer=buffer, file_id=file_id)
        if index_type is IndexType.RTREE:
            return RTreeIndex(column=column, buffer=buffer, file_id=file_id)
        return InvertedIndex(column=column, buffer=buffer, file_id=file_id)

    def _buffer_for(self, storage: StorageEngine) -> Any:
        from core.buffer.lru_buffer import LRUBufferManager

        buffer = self._buffers.get(id(storage))
        if buffer is None:
            buffer = LRUBufferManager(storage, capacity=self._buffer_capacity)
            self._buffers[id(storage)] = buffer
        return buffer

    def _table_and_column(self, schema: Schema) -> tuple[str, str]:
        if isinstance(schema, dict):
            return str(schema.get("table", "t")), str(schema.get("column", "col"))
        return "t", "col"
