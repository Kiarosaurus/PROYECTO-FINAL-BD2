from __future__ import annotations

import time
from typing import Any

from core.metrics import IOStats
from core.ports.storage import StorageEngine
from indices.ports import HybridPredicate

from query.explain import build_explain
from query.fusion import RankFusion, ReciprocalRankFusion
from query.ports import Executor
from query.plan_types import PlanOp, QueryPlan, ResultSet
from query.index_factory import IndexFactory, IndexType


# Ejecuta los planes usando la fábrica de índices
class QueryExecutor(Executor):

    def __init__(
        self,
        factory: IndexFactory,
        storage: StorageEngine,
        fusion: RankFusion | None = None,
    ) -> None:
        self._factory = factory
        self._storage = storage
        # Estrategia para combinar los rankings de una búsqueda combinada
        self._fusion = fusion if fusion is not None else ReciprocalRankFusion()
        # Columnas de cada tabla creada
        self._tables: dict[str, list[str]] = {}
        # Índice guardado por tabla y columna
        self._indexes: dict[tuple[str, str], Any] = {}
        # Nombre del tipo de índice creado por tabla y columna
        self._index_types: dict[tuple[str, str], str] = {}
        # Filas insertadas por tabla para recuperar el registro completo
        self._table_rows: dict[str, list[dict]] = {}

    def execute(self, plan: QueryPlan) -> ResultSet:
        start = time.perf_counter()
        stats = self._storage.stats()
        reads, writes, allocs = stats.disk_reads, stats.disk_writes, stats.pages_allocated
        result = self._dispatch(plan)
        # El costo de la consulta es lo que crecieron los contadores del storage
        stats = self._storage.stats()
        result.io = IOStats(
            disk_reads=stats.disk_reads - reads,
            disk_writes=stats.disk_writes - writes,
            pages_allocated=stats.pages_allocated - allocs,
        )
        result.elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        # El índice real usado pisa la sugerencia del planner
        if result.index_type is None:
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
        self._table_rows.pop(plan.table, None)
        for key in [k for k in self._indexes if k[0] == plan.table]:
            del self._indexes[key]
            self._index_types.pop(key, None)
        return ResultSet()

    def _create_index(self, plan: QueryPlan) -> ResultSet:
        index_type = IndexType.from_name(plan.index_type)
        schema = {
            "table": plan.table,
            "column": plan.columns[0],
            "columns": self._tables.get(plan.table, []),
        }
        vocabulary = plan.index_options.get("vocabulary")
        if vocabulary is not None:
            schema["vocabulary_limit"] = self._positive_int(vocabulary, "vocabulary")
        index = self._factory.create(index_type, schema, self._storage)
        self._indexes[(plan.table, plan.columns[0])] = index
        self._index_types[(plan.table, plan.columns[0])] = index_type.value
        return ResultSet()

    # Valida que la opción del índice sea un entero mayor que cero
    def _positive_int(self, value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"la opción {name} requiere un entero positivo, se recibió {value!r}")
        return value

    def _insert(self, plan: QueryPlan) -> ResultSet:
        cols = plan.columns or self._tables.get(plan.table, [])
        affected = 0
        for row in plan.rows:
            record = dict(zip(cols, row))
            self._table_rows.setdefault(plan.table, []).append(record)
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
        pred = plan.predicate
        if isinstance(pred, HybridPredicate):
            return self._hybrid_select(plan, pred)
        if pred is not None:
            key = (plan.table, pred.column)
            index = self._indexes.get(key)
        else:
            key, index = self._any_index(plan.table)
        records: list[Any] = []
        if index is not None:
            result = index.search(pred, plan.k)
            records = result.records
        columns, rows = self._project(plan, records)
        return ResultSet(columns=columns, rows=rows, index_type=self._index_types.get(key))

    # Corre las dos búsquedas de la consulta combinada
    # Junta los dos rankings en uno solo
    def _hybrid_select(self, plan: QueryPlan, pred: HybridPredicate) -> ResultSet:
        media_index = self._require_index(plan.table, pred.column)
        text_index = self._require_index(plan.table, pred.text.column)
        # Se piden más candidatos que k para que la fusión tenga de dónde elegir
        fetch = pred.k * 3
        media_result = media_index.search(pred.media, fetch)
        text_result = text_index.search(pred.text, fetch)
        visual = [(str(key), float(score)) for key, score in media_result.records]
        text, records_by_key = self._text_ranking(pred.column, text_result)
        fused = self._fusion.fuse([visual, text], k=pred.k)
        visual_scores = dict(visual)
        text_scores = dict(text)
        table_cols = self._tables.get(plan.table, [])
        wanted = table_cols if plan.columns == ["*"] else list(plan.columns)
        # Registro completo por clave para llenar los hits que solo vinieron del otro lado
        base_by_key = {
            str(record[pred.column]): record
            for record in self._table_rows.get(plan.table, [])
            if pred.column in record
        }
        rows = []
        for key, score in fused:
            record = records_by_key.get(key) or base_by_key.get(key)
            base = tuple(
                record.get(col) if record is not None else (key if col == pred.column else None)
                for col in wanted
            )
            rows.append(base + (score, visual_scores.get(key), text_scores.get(key)))
        columns = wanted + ["fused_score", "visual_score", "text_score"]
        return ResultSet(columns=columns, rows=rows, index_type="hybrid")

    # Falla con un mensaje claro si la columna no tiene índice
    def _require_index(self, table: str, column: str) -> Any:
        index = self._indexes.get((table, column))
        if index is None:
            raise ValueError(f"la búsqueda híbrida requiere un índice en {table}.{column}")
        return index

    # Arma el ranking de texto con la columna de media como clave
    def _text_ranking(self, media_column: str, result: Any):
        ranking: list[tuple[str, float]] = []
        records_by_key: dict[str, Any] = {}
        scores = result.scores or []
        for position, record in enumerate(result.records):
            if not isinstance(record, dict) or media_column not in record:
                continue
            key = str(record[media_column])
            score = float(scores[position]) if position < len(scores) else 0.0
            ranking.append((key, score))
            records_by_key[key] = record
        return ranking, records_by_key

    # Busca cualquier índice de la tabla para un escaneo sin filtro
    def _any_index(self, table: str) -> tuple[tuple[str, str] | None, Any]:
        for key, index in self._indexes.items():
            if key[0] == table:
                return key, index
        return None, None

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
