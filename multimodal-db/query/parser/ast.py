from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Una columna con su nombre y su tipo
@dataclass
class ColumnDef:
    name: str
    type: str


# Base de las condiciones de un WHERE
@dataclass
class Condition:
    column: str


# Compara una columna con un valor
@dataclass
class Comparison(Condition):
    op: str
    value: Any


# Pide valores entre dos límites
@dataclass
class Between(Condition):
    low: Any
    high: Any


# Busca los más parecidos a un vector o a un archivo
@dataclass
class KnnCondition(Condition):
    query: Any
    k: int


# Busca dentro de una caja entre dos esquinas
@dataclass
class SpatialCondition(Condition):
    min_corner: list
    max_corner: list


# Busca los textos que más coinciden con los términos
@dataclass
class MatchCondition(Condition):
    terms: str
    k: int | None = None


# Combina la búsqueda por archivo parecido con la búsqueda de texto
@dataclass
class HybridCondition(Condition):
    media_file: str
    text_column: str
    terms: str
    k: int


# Base de todas las sentencias
@dataclass
class Statement:
    pass


# Crea una tabla nueva
@dataclass
class CreateTable(Statement):
    table: str
    columns: list[ColumnDef] = field(default_factory=list)


# Borra una tabla
@dataclass
class DropTable(Statement):
    table: str


# Crea un índice sobre una columna
@dataclass
class CreateIndex(Statement):
    table: str
    column: str
    index_type: str


# Mete filas en una tabla
@dataclass
class Insert(Statement):
    table: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)


# Borra filas que cumplen una condición
@dataclass
class Delete(Statement):
    table: str
    where: Condition | None = None


# Pide filas de una tabla
@dataclass
class Select(Statement):
    table: str
    columns: list[str] = field(default_factory=list)
    where: Condition | None = None
    limit: int | None = None
