from __future__ import annotations

import json

from core.ports.buffer import BufferManager
from service.dto import ColumnSpec, IndexInfo, Schema, TableInfo

SNAPSHOT_PAGE_SIZE = 4096


# Guarda las tablas y sus índices
# Con buffer el contenido sobrevive a los reinicios
class Catalog:

    def __init__(self, buffer: BufferManager | None = None, file_id: str = "catalog") -> None:
        self._tables: dict[str, TableInfo] = {}
        self._buffer = buffer
        self._file_id = file_id
        self._load_snapshot()

    # Registra una tabla nueva
    def create_table(self, schema: Schema) -> TableInfo:
        if schema.table in self._tables:
            raise ValueError(f"la tabla ya existe: {schema.table}")
        info = TableInfo(name=schema.table, schema=schema)
        self._tables[schema.table] = info
        self._persist_snapshot()
        return info

    # Quita una tabla
    def drop_table(self, table: str) -> None:
        self._require(table)
        del self._tables[table]
        self._persist_snapshot()

    # Anota un índice sobre una columna existente
    def add_index(
        self,
        table: str,
        column: str,
        index_type: str,
        options: dict | None = None,
    ) -> IndexInfo:
        info = self._require(table)
        if info.schema.type_of(column) is None:
            raise ValueError(f"la columna no existe: {column}")
        index = IndexInfo(column=column, index_type=index_type, options=dict(options or {}))
        info.indexes.append(index)
        self._persist_snapshot()
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

    # Pasa las tablas a un formato simple para guardarlas
    def _encode_tables(self) -> dict:
        return {
            name: {
                "columns": [{"name": col.name, "type": col.type} for col in info.schema.columns],
                "indexes": [
                    {"column": ix.column, "index_type": ix.index_type, "options": ix.options}
                    for ix in info.indexes
                ],
            }
            for name, info in self._tables.items()
        }

    # Reconstruye las tablas desde el formato guardado
    def _decode_tables(self, payload: dict) -> dict[str, TableInfo]:
        tables: dict[str, TableInfo] = {}
        for name, data in payload.items():
            schema = Schema(
                table=name,
                columns=[ColumnSpec(name=col["name"], type=col["type"]) for col in data["columns"]],
            )
            indexes = [
                IndexInfo(
                    column=ix["column"],
                    index_type=ix["index_type"],
                    options=dict(ix.get("options") or {}),
                )
                for ix in data["indexes"]
            ]
            tables[name] = TableInfo(name=name, schema=schema, indexes=indexes)
        return tables

    def _persist_snapshot(self) -> None:
        if self._buffer is None:
            return
        state = json.dumps(self._encode_tables(), separators=(",", ":")).encode("utf-8")
        # El estado se parte en páginas en lugar de un solo bloque grande
        pages = [
            state[start:start + SNAPSHOT_PAGE_SIZE]
            for start in range(0, len(state), SNAPSHOT_PAGE_SIZE)
        ]
        metadata = {"version": 1, "state_page_count": len(pages)}
        self._write_page(0, json.dumps(metadata, separators=(",", ":")).encode("utf-8"))
        for page_no, page in enumerate(pages, start=1):
            self._write_page(page_no, page)
        self._buffer.flush(self._file_id)

    def _load_snapshot(self) -> None:
        if self._buffer is None:
            return
        raw = bytes(self._buffer.get(self._file_id, 0).data)
        if not raw:
            return
        try:
            metadata = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        stream = bytearray()
        for page_no in range(1, metadata.get("state_page_count", 0) + 1):
            stream.extend(self._buffer.get(self._file_id, page_no).data)
        if not stream:
            return
        try:
            payload = json.loads(stream.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        self._tables = self._decode_tables(payload)

    def _write_page(self, page_no: int, data: bytes) -> None:
        page = self._buffer.get(self._file_id, page_no)
        page.data[:] = data
        page.dirty = True
