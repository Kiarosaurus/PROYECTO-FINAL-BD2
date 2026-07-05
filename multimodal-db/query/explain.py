from __future__ import annotations

from indices.ports import Predicate, PredicateKind
from query.plan_types import PlanOp, QueryPlan, ResultSet

# Nombre del escaneo según el tipo de búsqueda
_SCAN = {
    PredicateKind.EQUALITY: "Index Scan",
    PredicateKind.RANGE: "Index Range Scan",
    PredicateKind.KNN: "KNN Index Scan",
    PredicateKind.SPATIAL_RANGE: "Spatial Index Scan",
    PredicateKind.TEXT_MATCH: "Text Search Scan",
    PredicateKind.HYBRID: "Hybrid Fusion Scan",
}


# Acorta un valor para que la línea no crezca demasiado
def _fmt(value: object) -> str:
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) > 6:
            head = ", ".join(str(v) for v in items[:6])
            return f"[{head}, ...]"
        return "[" + ", ".join(str(v) for v in items) + "]"
    return str(value)


# Línea de tiempos y costo real al estilo de actual time
def _measured(result: ResultSet) -> str:
    io = result.io
    return (
        f"(actual time={result.elapsed_ms} ms"
        f"  rows={len(result.rows)}"
        f"  reads={io.disk_reads}"
        f"  writes={io.disk_writes})"
    )


# Describe la condición del predicado al estilo Index Cond
def _condition(pred: Predicate) -> str | None:
    kind = pred.kind
    if kind is PredicateKind.EQUALITY:
        return f"Index Cond: {pred.column} = {_fmt(pred.value)}"
    if kind is PredicateKind.RANGE:
        low = ">=" if pred.include_low else ">"
        high = "<=" if pred.include_high else "<"
        return (
            f"Index Cond: {pred.column} {low} {_fmt(pred.low)}"
            f" AND {pred.column} {high} {_fmt(pred.high)}"
        )
    if kind is PredicateKind.KNN:
        return f"Order By: {pred.column} <-> {_fmt(pred.query)}"
    if kind is PredicateKind.SPATIAL_RANGE:
        return (
            f"Index Cond: {pred.column} <@ "
            f"box({_fmt(pred.min_corner)}, {_fmt(pred.max_corner)})"
        )
    if kind is PredicateKind.TEXT_MATCH:
        return f"Index Cond: {pred.column} @@ '{pred.terms}'"
    return None


# Cuántas filas cambió una operación de escritura
def _affected(result: ResultSet) -> int | None:
    if result.columns == ["affected"] and result.rows:
        return result.rows[0][0]
    return None


# Arma las líneas del plan como pares de profundidad y texto
def build_explain(plan: QueryPlan, result: ResultSet) -> list[tuple[int, str]]:
    if plan.op is PlanOp.SELECT:
        return _explain_select(plan, result)
    if plan.op is PlanOp.INSERT:
        return _explain_write("Insert", plan, result)
    if plan.op is PlanOp.DELETE:
        return _explain_write("Delete", plan, result)
    if plan.op is PlanOp.CREATE_TABLE:
        return [(0, f"Create Table {plan.table}  {_measured(result)}")]
    if plan.op is PlanOp.DROP_TABLE:
        return [(0, f"Drop Table {plan.table}  {_measured(result)}")]
    if plan.op is PlanOp.CREATE_INDEX:
        root = f"Create Index using {plan.index_type} on {plan.table}"
        lines = [(0, f"{root}  {_measured(result)}")]
        if plan.columns:
            lines.append((1, f"Column: {plan.columns[0]}"))
        return lines
    return [(0, f"{plan.op.name}  {_measured(result)}")]


def _explain_select(plan: QueryPlan, result: ResultSet) -> list[tuple[int, str]]:
    pred = plan.predicate
    if plan.index_type and pred is not None:
        scan = _SCAN.get(pred.kind, "Index Scan")
        root = f"{scan} using {plan.index_type} on {plan.table}"
    elif plan.index_type:
        root = f"Index Scan using {plan.index_type} on {plan.table}"
    else:
        root = f"Seq Scan on {plan.table}"
    lines = [(0, f"{root}  {_measured(result)}")]
    if pred is not None and pred.kind is PredicateKind.HYBRID:
        # Cada rama de la fusión se muestra como un escaneo hijo
        lines.append((1, f"Branch: {_SCAN[pred.media.kind]} on {pred.media.column}"))
        lines.append((2, _condition(pred.media)))
        lines.append((1, f"Branch: {_SCAN[pred.text.kind]} on {pred.text.column}"))
        lines.append((2, _condition(pred.text)))
    elif pred is not None:
        cond = _condition(pred)
        if cond is not None:
            lines.append((1, cond))
    if plan.k is not None:
        lines.append((1, f"Limit: {plan.k}"))
    if plan.index_type and pred is not None:
        lines.append(
            (1, f"Planner: predicado {pred.kind.name} usa índice {plan.index_type}")
        )
    return lines


def _explain_write(
    label: str, plan: QueryPlan, result: ResultSet
) -> list[tuple[int, str]]:
    lines = [(0, f"{label} on {plan.table}  {_measured(result)}")]
    affected = _affected(result)
    if affected is not None:
        lines.append((1, f"Tuples: {affected}"))
    pred = plan.predicate
    if pred is not None:
        cond = _condition(pred)
        if cond is not None:
            lines.append((1, cond))
    return lines
