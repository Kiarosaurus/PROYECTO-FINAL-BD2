from __future__ import annotations

from typing import Any

from core.metrics import IOStats

from query.ports import Parser, Planner, Executor
from query.plan_types import ResultSet


# Atiende las consultas de una sola conexión
class Session:

    def __init__(self, parser: Parser, planner: Planner, executor: Executor) -> None:
        self._parser = parser
        self._planner = planner
        self._executor = executor
        # Costo de I/O acumulado en la conexión
        self._io = IOStats()
        # Texto de la última consulta
        self._last_sql: str | None = None

    # Corre una consulta de principio a fin
    def execute(self, sql: str) -> ResultSet:
        self._last_sql = sql
        ast = self._parser.parse(sql)
        plan = self._planner.plan(ast, None)
        result = self._executor.execute(plan)
        self._io.merge(result.io)
        return result

    @property
    def io(self) -> IOStats:
        return self._io

    @property
    def last_sql(self) -> str | None:
        return self._last_sql

    # Pone en cero las métricas de la conexión
    def reset_metrics(self) -> None:
        self._io.reset()
