from __future__ import annotations

import time
from typing import Any

from core.metrics import IOStats
from core.ports.storage import StorageEngine

from query.explain import build_explain
from query.ports import Executor
from query.plan_types import PlanOp, QueryPlan, ResultSet
from query.index_factory import IndexFactory, IndexType


# Ejecuta los planes usando la fábrica de índices
class QueryExecutor(Executor):

    def __init__(self, factory: IndexFactory, storage: StorageEngine) -> None:
        self._factory = factory
        self._storage = storage
        # Columnas de cada tabla creada
        self._tables: dict[str, list[str]] = {}
        # Índice guardado por tabla y columna
        self._indexes: dict[tuple[str, str], Any] = {}

    def execute(self, plan: QueryPlan) -> ResultSet:
        start = time.perf_counter()
        result = self._dispatch(plan)
        result.elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        result.index_type = plan.index_type
        if plan.predicate is not None:
            result.predicate_kind = plan.predicate.kind.name.lower()
        result.explain = build_explain(plan, result)
        return result

    def _dispatch(self, plan: QueryPlan) -> ResultSet:
        if plan.op is PlanOp.CREATE_TABLE:
            return self._create_table(plan)
        if plan.op is PlanOp.DROP_TABLE:
            return self._drop_table(plan)
        if plan.op is PlanOp.CREATE_INDEX:
            return self._create_index(plan)
        if plan.op is PlanOp.INSERT:
            return self._insert(plan)
        if plan.op is PlanOp.DELETE:
            return self._delete(plan)
        if plan.op is PlanOp.SELECT:
            return self._select(plan)
        raise ValueError(f"operación no soportada: {plan.op.name}")

    def _create_table(self, plan: QueryPlan) -> ResultSet:
        self._tables[plan.table] = list(plan.columns)
        return ResultSet()

    def _drop_table(self, plan: QueryPlan) -> ResultSet:
        self._tables.pop(plan.table, None)
        for key in [k for k in self._indexes if k[0] == plan.table]:
            del self._indexes[key]
        return ResultSet()

    def _create_index(self, plan: QueryPlan) -> ResultSet:
        index_type = IndexType.from_name(plan.index_type)
        schema = self._tables.get(plan.table, [])
        index = self._factory.create(index_type, schema, self._storage)
        self._indexes[(plan.table, plan.columns[0])] = index
        return ResultSet()

    def _insert(self, plan: QueryPlan) -> ResultSet:
        cols = plan.columns or self._tables.get(plan.table, [])
        affected = 0
        for row in plan.rows:
            record = dict(zip(cols, row))
            for (table, column), index in self._indexes.items():
                if table == plan.table and column in record:
                    index.insert(record[column], record)
            affected += 1
        return self._count_result(affected)

    def _delete(self, plan: QueryPlan) -> ResultSet:
        affected = 0
        pred = plan.predicate
        if pred is not None:
            index = self._indexes.get((plan.table, pred.column))
            if index is not None and hasattr(pred, "value"):
                affected = index.delete(pred.value).affected
        return self._count_result(affected)

    # Arma un resultado que solo informa cuántas filas cambiaron
    def _count_result(self, affected: int) -> ResultSet:
        return ResultSet(columns=["affected"], rows=[(affected,)])

    def _select(self, plan: QueryPlan) -> ResultSet:
        io = IOStats()
        pred = plan.predicate
        if pred is not None:
            index = self._indexes.get((plan.table, pred.column))
        else:
            index = self._any_index(plan.table)
        records: list[Any] = []
        if index is not None:
            result = index.search(pred, plan.k)
            records = result.records
            io.merge(result.io)
        columns, rows = self._project(plan, records)
        return ResultSet(columns=columns, rows=rows, io=io)

    # Busca cualquier índice de la tabla para un escaneo sin filtro
    def _any_index(self, table: str) -> Any:
        for (name, _column), index in self._indexes.items():
            if name == table:
                return index
        return None

    # Deja solo las columnas pedidas en cada fila
    def _project(self, plan: QueryPlan, records: list[Any]):
        table_cols = self._tables.get(plan.table, [])
        wanted = table_cols if plan.columns == ["*"] else plan.columns
        rows = []
        for record in records:
            if isinstance(record, dict):
                rows.append(tuple(record.get(col) for col in wanted))
            else:
                rows.append((record,))
        return wanted, rows
