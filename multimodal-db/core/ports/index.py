from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from core.metrics import OperationResult

# Alias provisionales resueltos por las capas superiores
Key = Any
Record = Any
Predicate = Any


# Puerto de la capa de acceso
# Toda estructura de acceso expone la misma firma común
# Cubre construcción, inserción, búsqueda y borrado
class Index(ABC):

    @abstractmethod
    def build(self, records: Iterable[Record]) -> OperationResult:
        ...

    @abstractmethod
    def insert(self, key: Key, record: Record) -> OperationResult:
        ...

    @abstractmethod
    def search(self, predicate: Predicate, k: int | None = None) -> OperationResult:
        ...

    @abstractmethod
    def delete(self, key: Key) -> OperationResult:
        ...
