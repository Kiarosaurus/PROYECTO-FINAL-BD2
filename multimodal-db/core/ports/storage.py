from __future__ import annotations

from abc import ABC, abstractmethod

from core.metrics import IOStats


# Puerto de salida hacia el almacenamiento físico
# Única frontera que toca el medio persistente, sea heap file o Postgres
# Trabaja siempre a nivel de página identificada por file_id y page_no
class StorageEngine(ABC):

    @abstractmethod
    def read_page(self, file_id: str, page_no: int) -> bytes:
        ...

    @abstractmethod
    def write_page(self, file_id: str, page_no: int, data: bytes) -> None:
        ...

    @abstractmethod
    def allocate_page(self, file_id: str) -> int:
        ...

    @abstractmethod
    def stats(self) -> IOStats:
        ...
