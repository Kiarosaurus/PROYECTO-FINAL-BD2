from __future__ import annotations

from pathlib import Path

import pytest

from core.storage.file_engine import FileStorageEngine
from query.executor import QueryExecutor
from query.index_factory import EngineIndexFactory
from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from service.catalog import Catalog
from service.session import Session


@pytest.fixture
def run(tmp_path: Path):
    parser = SqlParser()
    planner = QueryPlanner()
    executor = QueryExecutor(EngineIndexFactory(), FileStorageEngine(tmp_path))

    def _run(sql: str):
        return executor.execute(planner.plan(parser.parse(sql)))

    return _run


def test_sql_pipeline_with_real_hash_index_and_file_storage(run) -> None:
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    insert = run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')
    found = run("SELECT path FROM img WHERE id = 2")

    assert insert.rows == [(2,)]
    assert found.rows == [("b.jpg",)]
    assert insert.io.disk_writes > 0


def test_sql_range_query_uses_real_bplus_index(run) -> None:
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING bplus")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")')

    result = run("SELECT id FROM img WHERE id BETWEEN 2 AND 3")

    assert result.rows == [(2,), (3,)]
    assert result.index_type == "bplus"
    assert result.predicate_kind == "range"


def test_sql_select_without_where_scans_real_index(run) -> None:
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING bplus")
    run('INSERT INTO img (id, path) VALUES (2, "b.jpg"), (1, "a.jpg")')

    result = run("SELECT id FROM img")

    assert result.rows == [(1,), (2,)]


def test_sql_spatial_query_uses_real_rtree(run) -> None:
    run("CREATE TABLE places (id INT, point VECTOR)")
    run("CREATE INDEX ON places (point) USING rtree")
    executor_result = run("SELECT id FROM places WHERE WITHIN(point, [0, 0], [10, 10])")

    assert executor_result.rows == []
    assert executor_result.index_type == "rtree"


def test_session_with_catalog_makes_planner_respect_created_index(tmp_path: Path) -> None:
    executor = QueryExecutor(EngineIndexFactory(), FileStorageEngine(tmp_path))
    session = Session(SqlParser(), QueryPlanner(), executor, Catalog())

    session.execute("CREATE TABLE img (id INT, path TEXT)")
    session.execute("CREATE INDEX ON img (id) USING bplus")
    session.execute('INSERT INTO img (id, path) VALUES (1, "a.jpg")')
    result = session.execute("SELECT path FROM img WHERE id = 1")

    assert result.rows == [("a.jpg",)]
    # Sin catálogo el planner sugiere hash, con catálogo respeta el bplus creado
    assert result.index_type == "bplus"


def test_sql_delete_removes_from_real_index(run) -> None:
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')

    deleted = run("DELETE FROM img WHERE id = 1")
    remaining = run("SELECT id FROM img")

    assert deleted.rows == [(1,)]
    assert remaining.rows == [(2,)]
