from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.metrics import IOStats


@dataclass
class Page:
    file_id: str
    page_no: int
    data: bytearray
    # Marca si el contenido difiere del que está en disco
    dirty: bool = False
    # Número de clientes que mantienen fija la página en memoria
    pin_count: int = 0


# Cache de páginas entre los índices y el StorageEngine
# Único componente que invoca al StorageEngine
# Único lugar donde se contabilizan los accesos a disco
class BufferManager(ABC):

    @abstractmethod
    def get(self, file_id: str, page_no: int) -> Page:
        ...

    @abstractmethod
    def pin(self, page: Page) -> None:
        ...

    @abstractmethod
    def unpin(self, page: Page) -> None:
        ...

    @abstractmethod
    def flush(self, file_id: str | None = None) -> None:
        ...

    @abstractmethod
    def allocate_page(self, file_id: str) -> int:
        ...

    @abstractmethod
    def stats(self) -> IOStats:
        ...
