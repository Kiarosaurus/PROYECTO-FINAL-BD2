from __future__ import annotations

import os
import uuid

import psycopg2
import pytest

from core.metrics import IOStats
from core.storage.postgres_engine import PostgresStorageEngine

DEFAULT_DSN = "postgresql://mmdb:mmdb@localhost:5432/multimodal"


def _dsn() -> str:
    return os.environ.get("POSTGRES_TEST_DSN", DEFAULT_DSN)


def _connect_or_skip(dsn: str):
    try:
        return psycopg2.connect(dsn)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"no hay un Postgres disponible en {dsn}: {exc}")


@pytest.fixture
def storage():
    dsn = _dsn()
    raw = _connect_or_skip(dsn)
    raw.autocommit = True
    engine = PostgresStorageEngine(dsn)
    file_id = f"it_{uuid.uuid4().hex}"
    yield engine, raw, file_id
    with raw.cursor() as cur:
        cur.execute("DELETE FROM engine.page WHERE file_id = %s", (file_id,))
    raw.close()
    engine.close()


def test_write_then_read_roundtrip_against_real_postgres(storage) -> None:
    engine, _raw, file_id = storage
    engine.write_page(file_id, 0, b"hola postgres")
    assert engine.read_page(file_id, 0) == b"hola postgres"


def test_allocate_page_persists_across_new_connections(storage) -> None:
    engine, _raw, file_id = storage
    first = engine.allocate_page(file_id)
    second = engine.allocate_page(file_id)
    assert second == first + 1

    reconnected = PostgresStorageEngine(_dsn())
    third = reconnected.allocate_page(file_id)
    assert third == second + 1
    reconnected.close()


def test_stats_are_tracked_against_real_postgres(storage) -> None:
    engine, _raw, file_id = storage
    engine.write_page(file_id, 0, b"x")
    engine.read_page(file_id, 0)
    stats = engine.stats()
    assert isinstance(stats, IOStats)
    assert stats.disk_writes >= 1
    assert stats.disk_reads >= 1


# El schema engine depende de pgvector para los histogramas de multimedia
def test_pgvector_extension_is_installed(storage) -> None:
    _engine, raw, _file_id = storage
    with raw.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        assert cur.fetchone() is not None, "la extension vector no esta instalada en la base de datos"
