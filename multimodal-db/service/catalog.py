from __future__ import annotations

from service.dto import IndexInfo, Schema, TableInfo


# Guarda las tablas y sus índices
class Catalog:

    def __init__(self) -> None:
        self._tables: dict[str, TableInfo] = {}

    # Registra una tabla nueva
    def create_table(self, schema: Schema) -> TableInfo:
        if schema.table in self._tables:
            raise ValueError(f"la tabla ya existe: {schema.table}")
        info = TableInfo(name=schema.table, schema=schema)
        self._tables[schema.table] = info
        return info

    # Quita una tabla
    def drop_table(self, table: str) -> None:
        self._require(table)
        del self._tables[table]

    # Anota un índice sobre una columna existente
    def add_index(self, table: str, column: str, index_type: str) -> IndexInfo:
        info = self._require(table)
        if info.schema.type_of(column) is None:
            raise ValueError(f"la columna no existe: {column}")
        index = IndexInfo(column=column, index_type=index_type)
        info.indexes.append(index)
        return index

    # Devuelve los datos de una tabla
    def get_table(self, table: str) -> TableInfo:
        return self._require(table)

    # Dice si la tabla está registrada
    def has_table(self, table: str) -> bool:
        return table in self._tables

    # Lista los nombres de las tablas
    def list_tables(self) -> list[str]:
        return list(self._tables.keys())

    def _require(self, table: str) -> TableInfo:
        if table not in self._tables:
            raise KeyError(f"la tabla no existe: {table}")
        return self._tables[table]
