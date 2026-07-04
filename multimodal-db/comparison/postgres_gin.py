from __future__ import annotations

import time
import tracemalloc
from typing import Any, Literal

import psycopg2

from comparison.ports import BenchmarkResult, ComparisonEngine


class PostgresGINEngine(ComparisonEngine):
    # Conecta al schema compare para búsqueda full-text con GIN
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _conn(self):
        return psycopg2.connect(self._dsn)

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
        tracemalloc.start()
        t0 = time.perf_counter()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (str(q),))
            rows = cur.fetchall()
        latency_ms = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return BenchmarkResult(
            records=rows,
            latency_ms=round(latency_ms, 3),
            memory_bytes=peak,
        )
