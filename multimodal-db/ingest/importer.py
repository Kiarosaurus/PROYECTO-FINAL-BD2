from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ingest.ports import DatasetLoader
from service.session import Session


# Resumen de una importación terminada
@dataclass
class ImportReport:
    table: str
    rows_inserted: int = 0
    batches: int = 0
    indexes: list[str] = field(default_factory=list)


_WHITESPACE = re.compile(r"\s+")


# Lleva un dataset completo al engine usando solo SQL
class DatasetImporter:

    def __init__(self, session: Session, batch_size: int = 200) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size debe ser positivo")
        self._session = session
        self._batch_size = batch_size

    # Crea la tabla y sus índices y luego inserta las filas por lotes
    def run(self, loader: DatasetLoader, limit: int | None = None) -> ImportReport:
        table = loader.table_name()
        columns = loader.columns()
        defs = ", ".join(f"{col.name} {col.type}" for col in columns)
        self._session.execute(f"CREATE TABLE {table} ({defs})")
        report = ImportReport(table=table)
        # Los índices van primero para que reciban cada fila insertada
        for index in loader.indexes():
            self._session.execute(
                f"CREATE INDEX ON {table} ({index.column}) USING {index.index_type}"
            )
            report.indexes.append(f"{index.column}:{index.index_type}")
        names = ", ".join(col.name for col in columns)
        batch: list[str] = []
        for count, row in enumerate(loader.rows()):
            if limit is not None and count >= limit:
                break
            batch.append(self._format_row(row))
            if len(batch) >= self._batch_size:
                self._flush(table, names, batch, report)
        if batch:
            self._flush(table, names, batch, report)
        return report

    def _flush(self, table: str, names: str, batch: list[str], report: ImportReport) -> None:
        values = ", ".join(batch)
        self._session.execute(f"INSERT INTO {table} ({names}) VALUES {values}")
        report.rows_inserted += len(batch)
        report.batches += 1
        batch.clear()

    def _format_row(self, row: tuple) -> str:
        return "(" + ", ".join(self._format_value(value) for value in row) + ")"

    # Escribe cada valor como literal SQL del engine
    def _format_value(self, value: Any) -> str:
        if isinstance(value, str):
            return '"' + self._clean_text(value) + '"'
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(self._format_value(item) for item in value) + "]"
        if isinstance(value, bool):
            raise TypeError("bool no tiene literal SQL en el engine")
        if isinstance(value, float):
            return f"{value:.12f}"
        if isinstance(value, int):
            return str(value)
        raise TypeError(f"valor no soportado en la fila: {type(value).__name__}")

    # Deja el texto en una sola línea y sin comillas conflictivas
    def _clean_text(self, text: str) -> str:
        flat = _WHITESPACE.sub(" ", text).strip()
        return flat.replace("\\", " ").replace('"', "'")
