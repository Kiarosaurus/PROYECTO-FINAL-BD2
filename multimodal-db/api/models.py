from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# Lo que el cliente manda para correr una consulta
class QueryRequest(BaseModel):
    sql: str


# Costo de I/O de la consulta
class IOStatsModel(BaseModel):
    disk_reads: int = 0
    disk_writes: int = 0
    pages_allocated: int = 0


# Lo que el servidor responde con los resultados
class QueryResponse(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    io: IOStatsModel = Field(default_factory=IOStatsModel)
    index_type: str | None = None
    predicate_kind: str | None = None
    elapsed_ms: float = 0.0


# Lo que el servidor responde cuando algo falla
class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
