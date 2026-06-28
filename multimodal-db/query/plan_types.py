from __future__ import annotations

from dataclasses import dataclass, field

from enum import Enum, auto

from core.metrics import IOStats
from indices.ports import Predicate


# Cosas que se pueden pedir
class PlanOp(Enum):
    CREATE_TABLE = auto()
    DROP_TABLE = auto()
    CREATE_INDEX = auto()
    INSERT = auto()
    DELETE = auto()
    SELECT = auto()


# Plan listo para ejecutar
@dataclass
class QueryPlan:
    op: PlanOp
    table: str
    columns: list[str] = field(default_factory=list)
    # Tipo de cada columna al crear una tabla
    column_types: list[str] = field(default_factory=list)
    # Condición para filtrar filas
    predicate: Predicate | None = None
    # Filas a insertar
    rows: list[tuple] = field(default_factory=list)
    # Cuántos resultados pedir
    k: int | None = None
    # Tipo de índice a crear
    index_type: str | None = None


# Resultado que sale del executor
@dataclass
class ResultSet:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    io: IOStats = field(default_factory=IOStats)
    # Índice que se usó para resolver la consulta
    index_type: str | None = None
    # Familia de búsqueda del predicado
    predicate_kind: str | None = None
    # Tiempo de ejecución en milisegundos
    elapsed_ms: float = 0.0
