from __future__ import annotations

from typing import Any

from indices.ports import EqualityPredicate, RangePredicate

from query.ports import Planner
from query.plan_types import PlanOp, QueryPlan
from query.parser import ast as A

# Pasa cada signo de comparación a un rango con un lado abierto
_RANGE_OPS = {
    "<": dict(low=None, include_low=True, include_high=False),
    "<=": dict(low=None, include_low=True, include_high=True),
    ">": dict(high=None, include_low=False, include_high=True),
    ">=": dict(high=None, include_low=True, include_high=True),
}


# Arma el QueryPlan a partir del AST
class QueryPlanner(Planner):

    # Convierte una condición simple en un Predicate
    def _to_predicate(self, cond: Any):
        if cond is None:
            return None
        if isinstance(cond, A.Comparison):
            if cond.op == "=":
                return EqualityPredicate(column=cond.column, value=cond.value)
            if cond.op in _RANGE_OPS:
                opts = dict(_RANGE_OPS[cond.op])
                opts.setdefault("low", cond.value)
                opts.setdefault("high", cond.value)
                return RangePredicate(column=cond.column, **opts)
            raise ValueError(f"operador no soportado: {cond.op}")
        if isinstance(cond, A.Between):
            return RangePredicate(column=cond.column, low=cond.low, high=cond.high)
        raise ValueError(f"condición no soportada: {type(cond).__name__}")

    def plan(self, ast: Any, catalog: Any = None) -> QueryPlan:
        if isinstance(ast, A.CreateTable):
            return QueryPlan(
                op=PlanOp.CREATE_TABLE,
                table=ast.table,
                columns=[col.name for col in ast.columns],
                column_types=[col.type for col in ast.columns],
            )
        if isinstance(ast, A.DropTable):
            return QueryPlan(op=PlanOp.DROP_TABLE, table=ast.table)
        if isinstance(ast, A.CreateIndex):
            return QueryPlan(
                op=PlanOp.CREATE_INDEX,
                table=ast.table,
                columns=[ast.column],
                index_type=ast.index_type,
            )
        if isinstance(ast, A.Insert):
            return QueryPlan(
                op=PlanOp.INSERT,
                table=ast.table,
                columns=list(ast.columns),
                rows=list(ast.rows),
            )
        if isinstance(ast, A.Delete):
            return QueryPlan(
                op=PlanOp.DELETE,
                table=ast.table,
                predicate=self._to_predicate(ast.where),
            )
        if isinstance(ast, A.Select):
            return QueryPlan(
                op=PlanOp.SELECT,
                table=ast.table,
                columns=list(ast.columns),
                predicate=self._to_predicate(ast.where),
                k=ast.limit,
            )
        raise ValueError(f"sentencia no soportada: {type(ast).__name__}")
