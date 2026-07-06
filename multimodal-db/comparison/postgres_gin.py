from __future__ import annotations

import time
from typing import Any, Literal

import psycopg2

from comparison.ports import BenchmarkResult, ComparisonEngine


class PostgresGINEngine(ComparisonEngine):
    # Conecta al schema compare para búsqueda full-text con GIN
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._connection = None

    # Reutiliza una sola conexión para no medir el costo de conectar
    def _conn(self):
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(self._dsn)
        return self._connection

    def load(self, dataset: list[dict]) -> None:
        # Inserta letras de canciones en la tabla de documentos
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE compare.documents RESTART IDENTITY")
            rows = [
                (i, entry["lyrics"])
                for i, entry in enumerate(dataset)
                if entry.get("lyrics")
            ]
            cur.executemany(
                "INSERT INTO compare.documents (id, body) VALUES (%s, %s)",
                rows,
            )
            conn.commit()

    def build_native_index(self, kind: Literal["GIN", "GiST", "HNSW", "IVFFlat"]) -> None:
        # El índice GIN ya existe desde init.sql
        if kind not in ("GIN", "GiST"):
            raise ValueError(f"PostgresGINEngine no soporta índice {kind}")
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("REINDEX INDEX compare.idx_documents_fts")
            conn.commit()

    def query(self, q: Any) -> BenchmarkResult:
        # q es el texto de búsqueda
        sql = """
            SELECT id, body, ts_rank(fts, query) AS rank
            FROM compare.documents, plainto_tsquery('english', %s) query
            WHERE fts @@ query
            ORDER BY rank DESC
            LIMIT 10
        """
        conn = self._conn()
        t0 = time.perf_counter()
        with conn, conn.cursor() as cur:
            cur.execute(sql, (str(q),))
            rows = cur.fetchall()
        latency_ms = (time.perf_counter() - t0) * 1000
        return BenchmarkResult(
            records=rows,
            latency_ms=round(latency_ms, 3),
        )
