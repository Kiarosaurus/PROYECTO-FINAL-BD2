from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IOStats:
    disk_reads: int = 0
    disk_writes: int = 0
    pages_allocated: int = 0

    def add_read(self, count: int = 1) -> None:
        self.disk_reads += count

    def add_write(self, count: int = 1) -> None:
        self.disk_writes += count

    def add_allocation(self, count: int = 1) -> None:
        self.pages_allocated += count

    def merge(self, other: "IOStats") -> None:
        self.disk_reads += other.disk_reads
        self.disk_writes += other.disk_writes
        self.pages_allocated += other.pages_allocated

    def reset(self) -> None:
        self.disk_reads = 0
        self.disk_writes = 0
        self.pages_allocated = 0


@dataclass
class OperationResult:
    success: bool = True
    # Tuplas devueltas por una operación de búsqueda o lectura
    records: list[Any] = field(default_factory=list)
    # Score de cada tupla devuelta cuando la búsqueda ordena por parecido
    scores: list[float] | None = None
    # Cantidad de tuplas afectadas por insert o delete
    affected: int = 0
    # Costo de I/O atribuible a esta operación
    io: IOStats = field(default_factory=IOStats)
    message: str | None = None

    @classmethod
    def failure(cls, message: str) -> "OperationResult":
        return cls(success=False, message=message)
