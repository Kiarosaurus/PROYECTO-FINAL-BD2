import pytest

from core.metrics import IOStats
from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from query.executor import QueryExecutor
from query.index_factory import IndexType
from tests.mocks import MockIndexFactory, MockStorageEngine


@pytest.fixture
def engine():
    parser = SqlParser()
    planner = QueryPlanner()
    factory = MockIndexFactory()
    executor = QueryExecutor(factory, MockStorageEngine())

    def run(sql):
        return executor.execute(planner.plan(parser.parse(sql)))

    return run, executor, factory


def test_create_index_uses_factory(engine):
    run, _executor, factory = engine
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    run("CREATE INDEX ON img (path) USING inverted")
    assert factory.created == [IndexType.HASH, IndexType.INVERTED]


def test_insert_affected(engine):
    run, _executor, _factory = engine
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    result = run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')
    assert result.columns == ["affected"]
    assert result.rows == [(2,)]


def test_insert_populates_index(engine):
    run, executor, _factory = engine
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')
    index = executor._indexes[("img", "id")]
    assert index.search(None).records == [
        {"id": 1, "path": "a.jpg"},
        {"id": 2, "path": "b.jpg"},
    ]


def test_select_projection(engine):
    run, _executor, _factory = engine
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')
    result = run("SELECT path FROM img")
    assert result.columns == ["path"]
    assert result.rows == [("a.jpg",), ("b.jpg",)]


def test_select_star_expands_columns(engine):
    run, _executor, _factory = engine
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg")')
    result = run("SELECT * FROM img")
    assert result.columns == ["id", "path"]
    assert result.rows == [(1, "a.jpg")]


def test_select_limit_slices(engine):
    run, _executor, _factory = engine
    run("CREATE TABLE img (id INT, path TEXT)")
    run("CREATE INDEX ON img (id) USING hash")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg"), (3, "c.jpg")')
    result = run("SELECT id FROM img LIMIT 1")
    assert result.rows == [(1,)]


def test_select_returns_iostats(engine):
    run, _executor, _factory = engine
    run("CREATE TABLE img (id INT)")
    run("CREATE INDEX ON img (id) USING hash")
    result = run("SELECT * FROM img")
    assert isinstance(result.io, IOStats)


def test_select_without_index_is_empty(engine):
    run, _executor, _factory = engine
    run("CREATE TABLE img (id INT, path TEXT)")
    run('INSERT INTO img (id, path) VALUES (1, "a.jpg")')
    result = run("SELECT * FROM img")
    assert result.rows == []


def test_delete_returns_affected_column(engine):
    run, _executor, _factory = engine
    run("CREATE TABLE img (id INT)")
    run("CREATE INDEX ON img (id) USING hash")
    result = run("DELETE FROM img WHERE id = 2")
    assert result.columns == ["affected"]


def test_drop_table_clears_indexes(engine):
    run, executor, _factory = engine
    run("CREATE TABLE img (id INT)")
    run("CREATE INDEX ON img (id) USING hash")
    run("DROP TABLE img")
    assert executor._tables == {}
    assert executor._indexes == {}
