from __future__ import annotations

import psycopg2

from core.metrics import IOStats
from core.ports.storage import StorageEngine


class PostgresStorageEngine(StorageEngine):

    def __init__(self, dsn: str) -> None:
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        self._stats = IOStats()

    def read_page(self, file_id: str, page_no: int) -> bytes:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT data FROM engine.page WHERE file_id = %s AND page_no = %s",
                (file_id, page_no),
            )
            row = cur.fetchone()
        self._stats.add_read()
        return bytes(row[0]) if row is not None else b""

    def write_page(self, file_id: str, page_no: int, data: bytes) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO engine.page (file_id, page_no, data)
                VALUES (%s, %s, %s)
                ON CONFLICT (file_id, page_no) DO UPDATE SET data = EXCLUDED.data
                """,
                (file_id, page_no, bytes(data)),
            )
        self._stats.add_write()

    # El propio MAX(page_no) hace de contador, sin necesitar una tabla aparte
    def allocate_page(self, file_id: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO engine.page (file_id, page_no, data)
                SELECT %s, COALESCE(MAX(page_no), -1) + 1, ''::bytea
                FROM engine.page WHERE file_id = %s
                RETURNING page_no
                """,
                (file_id, file_id),
            )
            page_no = cur.fetchone()[0]
        self._stats.add_allocation()
        return page_no

    def stats(self) -> IOStats:
        return self._stats

    def close(self) -> None:
        self._conn.close()
