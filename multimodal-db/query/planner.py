from __future__ import annotations

from typing import Any

from indices.ports import (
    EqualityPredicate,
    HybridPredicate,
    KnnPredicate,
    PredicateKind,
    RangePredicate,
    SpatialRangePredicate,
    TextMatchPredicate,
)

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

# Índice que mejor resuelve cada tipo de búsqueda
_INDEX_BY_KIND = {
    PredicateKind.EQUALITY: "hash",
    PredicateKind.RANGE: "bplus",
    PredicateKind.KNN: "knn",
    PredicateKind.SPATIAL_RANGE: "rtree",
    PredicateKind.TEXT_MATCH: "inverted",
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
        if isinstance(cond, A.KnnCondition):
            return KnnPredicate(column=cond.column, query=cond.query, k=cond.k)
        if isinstance(cond, A.SpatialCondition):
            return SpatialRangePredicate(
                column=cond.column,
                min_corner=cond.min_corner,
                max_corner=cond.max_corner,
            )
        if isinstance(cond, A.MatchCondition):
            return TextMatchPredicate(column=cond.column, terms=cond.terms, k=cond.k)
        if isinstance(cond, A.HybridCondition):
            return HybridPredicate(
                column=cond.column,
                media=KnnPredicate(column=cond.column, query=cond.media_file, k=cond.k),
                text=TextMatchPredicate(column=cond.text_column, terms=cond.terms, k=cond.k),
                k=cond.k,
            )
        raise ValueError(f"condición no soportada: {type(cond).__name__}")

    # Elige el índice adecuado para el predicado
    def _index_type_for(self, predicate: Any, table: str | None = None, catalog: Any = None):
        if predicate is None:
            return None
        # La búsqueda combinada se reporta con su propio nombre
        if predicate.kind is PredicateKind.HYBRID:
            return "hybrid"
        registered = self._registered_index(catalog, table, predicate.column)
        if registered is not None:
            return registered
        return _INDEX_BY_KIND.get(predicate.kind)

    # Si la columna ya tiene un índice creado, el catálogo manda
    def _registered_index(self, catalog: Any, table: str | None, column: str):
        if catalog is None or table is None or not catalog.has_table(table):
            return None
        for index in catalog.get_table(table).indexes:
            if index.column == column:
                return index.index_type
        return None

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
                index_options=dict(ast.options),
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
            predicate = self._to_predicate(ast.where)
            # El k propio del predicado tiene prioridad sobre el LIMIT
            k = ast.limit
            if isinstance(predicate, (KnnPredicate, HybridPredicate)):
                k = predicate.k
            elif isinstance(predicate, TextMatchPredicate) and predicate.k is not None:
                k = predicate.k
            return QueryPlan(
                op=PlanOp.SELECT,
                table=ast.table,
                columns=list(ast.columns),
                predicate=predicate,
                k=k,
                index_type=self._index_type_for(predicate, ast.table, catalog),
            )
        raise ValueError(f"sentencia no soportada: {type(ast).__name__}")
