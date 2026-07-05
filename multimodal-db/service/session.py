from __future__ import annotations

from typing import Any

from core.metrics import IOStats

from query.parser import ast as A
from query.ports import Parser, Planner, Executor
from query.plan_types import ResultSet
from service.dto import ColumnSpec, Schema


# Atiende las consultas de una sola conexión
class Session:

    def __init__(
        self,
        parser: Parser,
        planner: Planner,
        executor: Executor,
        catalog: Any | None = None,
    ) -> None:
        self._parser = parser
        self._planner = planner
        self._executor = executor
        self._catalog = catalog
        # Costo de I/O acumulado en la conexión
        self._io = IOStats()
        # Texto de la última consulta
        self._last_sql: str | None = None

    # Corre una consulta de principio a fin
    def execute(self, sql: str) -> ResultSet:
        self._last_sql = sql
        ast = self._parser.parse(sql)
        plan = self._planner.plan(ast, self._catalog)
        result = self._executor.execute(plan)
        self._io.merge(result.io)
        self._register_ddl(ast)
        return result

    # Mantiene el catálogo al día cuando cambian las tablas
    def _register_ddl(self, ast: Any) -> None:
        if self._catalog is None:
            return
        if isinstance(ast, A.CreateTable) and not self._catalog.has_table(ast.table):
            schema = Schema(
                table=ast.table,
                columns=[ColumnSpec(name=col.name, type=col.type) for col in ast.columns],
            )
            self._catalog.create_table(schema)
        elif isinstance(ast, A.DropTable) and self._catalog.has_table(ast.table):
            self._catalog.drop_table(ast.table)
        elif isinstance(ast, A.CreateIndex) and self._catalog.has_table(ast.table):
            info = self._catalog.get_table(ast.table)
            if info.schema.type_of(ast.column) is not None:
                self._catalog.add_index(ast.table, ast.column, ast.index_type)

    @property
    def io(self) -> IOStats:
        return self._io

    @property
    def last_sql(self) -> str | None:
        return self._last_sql

    # Pone en cero las métricas de la conexión
    def reset_metrics(self) -> None:
        self._io.reset()
