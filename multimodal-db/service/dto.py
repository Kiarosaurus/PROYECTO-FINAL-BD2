from __future__ import annotations

from dataclasses import dataclass, field


# Una columna con su nombre y su tipo
@dataclass
class ColumnSpec:
    name: str
    type: str


# Las columnas de una tabla
@dataclass
class Schema:
    table: str
    columns: list[ColumnSpec] = field(default_factory=list)

    # Lista solo los nombres de columna
    def column_names(self) -> list[str]:
        return [col.name for col in self.columns]

    # Devuelve el tipo de una columna
    def type_of(self, name: str) -> str | None:
        for col in self.columns:
            if col.name == name:
                return col.type
        return None


# Un índice creado sobre una columna
@dataclass
class IndexInfo:
    column: str
    index_type: str
    # Ajustes con los que se creó el índice
    options: dict = field(default_factory=dict)


# Datos de una tabla y sus índices
@dataclass
class TableInfo:
    name: str
    schema: Schema
    indexes: list[IndexInfo] = field(default_factory=list)
