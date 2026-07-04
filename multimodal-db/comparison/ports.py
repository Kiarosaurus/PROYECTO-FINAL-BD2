from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class BenchmarkResult:
    # Resultados devueltos por la consulta
    records: list[Any] = field(default_factory=list)
    # Tiempo de consulta en milisegundos
    latency_ms: float = 0.0
    # Uso de memoria en bytes
    memory_bytes: int = 0


# Motor de comparación contra índices nativos de PostgreSQL
class ComparisonEngine(ABC):

    # Carga el conjunto de datos en la base de datos
    @abstractmethod
    def load(self, dataset: list[dict]) -> None:
        ...

    # Construye el índice nativo indicado
    @abstractmethod
    def build_native_index(self, kind: Literal["GIN", "GiST", "HNSW", "IVFFlat"]) -> None:
        ...

    # Ejecuta una consulta y retorna resultados con métricas
    @abstractmethod
    def query(self, q: Any) -> BenchmarkResult:
        ...
