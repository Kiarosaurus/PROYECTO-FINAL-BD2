from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from comparison.ports import BenchmarkResult, ComparisonEngine


class PgVectorIVFEngine(ComparisonEngine):
    # Conecta al schema compare para KNN aproximado con IVFFlat
    def __init__(self, dsn: str, modality: str = "IMAGE") -> None:
        self._dsn = dsn
        self._modality = modality.upper()
        self._connection = None

    # Reutiliza una sola conexión para no medir el costo de conectar
    def _conn(self):
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(self._dsn)
        return self._connection

    def load(self, dataset: list[dict]) -> None:
        # Inserta histogramas como vectores en la tabla de medios
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM compare.media WHERE modality = %s",
                (self._modality,),
            )
            rows = [
                (i, self._modality, entry.get("path", ""), self._to_pg_vec(entry["vector"]))
                for i, entry in enumerate(dataset)
                if entry.get("vector") is not None
            ]
            execute_values(
                cur,
                "INSERT INTO compare.media (id, modality, path, feature_vec) VALUES %s",
                rows,
            )
            conn.commit()

    def build_native_index(self, kind: Literal["GIN", "GiST", "HNSW", "IVFFlat"]) -> None:
        if kind not in ("IVFFlat", "HNSW"):
            raise ValueError(f"PgVectorIVFEngine no soporta índice {kind}")
        with self._conn() as conn, conn.cursor() as cur:
            # ANALYZE asegura estadísticas actualizadas antes de reconstruir
            cur.execute("ANALYZE compare.media")
            if kind == "IVFFlat":
                cur.execute("REINDEX INDEX compare.idx_media_ivfflat")
            else:
                # El HNSW se crea recién acá porque init.sql solo trae el IVFFlat
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_media_hnsw "
                    "ON compare.media USING hnsw (feature_vec vector_cosine_ops)"
                )
                cur.execute("REINDEX INDEX compare.idx_media_hnsw")
            conn.commit()

    def query(self, q: Any) -> BenchmarkResult:
        # q es el histograma de la consulta como array numpy o lista
        vec_str = self._to_pg_vec(np.asarray(q, dtype=np.float32))
        sql = """
            SELECT id, path, feature_vec <=> %s::vector AS distance
            FROM compare.media
            WHERE modality = %s
            ORDER BY distance
            LIMIT 10
        """
        conn = self._conn()
        t0 = time.perf_counter()
        with conn, conn.cursor() as cur:
            cur.execute(sql, (vec_str, self._modality))
            rows = cur.fetchall()
        latency_ms = (time.perf_counter() - t0) * 1000
        return BenchmarkResult(
            records=rows,
            latency_ms=round(latency_ms, 3),
        )

    @staticmethod
    def _to_pg_vec(arr: np.ndarray) -> str:
        # Convierte un array a la representación de texto que acepta pgvector
        values = ",".join(f"{v:.6f}" for v in arr.tolist())
        return f"[{values}]"
