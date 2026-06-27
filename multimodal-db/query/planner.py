from __future__ import annotations

from typing import Any

from query.ports import Planner
from query.plan_types import PlanOp, QueryPlan
from query.parser import ast as A


# Arma el QueryPlan a partir del AST
class QueryPlanner(Planner):

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
        raise ValueError(f"sentencia no soportada: {type(ast).__name__}")
