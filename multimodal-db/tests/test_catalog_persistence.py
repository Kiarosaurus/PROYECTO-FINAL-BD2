from __future__ import annotations

from pathlib import Path

from core.buffer.lru_buffer import LRUBufferManager
from core.storage.file_engine import FileStorageEngine
from query.executor import QueryExecutor
from query.index_factory import EngineIndexFactory
from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from service.catalog import Catalog
from service.dto import ColumnSpec, Schema
from service.session import Session, rehydrate_executor


def _persistent_catalog(base_dir: Path) -> Catalog:
    storage = FileStorageEngine(base_dir)
    return Catalog(buffer=LRUBufferManager(storage))


def test_catalog_without_buffer_keeps_working_in_memory() -> None:
    catalog = Catalog()
    catalog.create_table(Schema(table="docs", columns=[ColumnSpec(name="id", type="INT")]))

    assert catalog.has_table("docs")


def test_catalog_snapshot_survives_a_second_instance(tmp_path: Path) -> None:
    first = _persistent_catalog(tmp_path)
    first.create_table(
        Schema(
            table="docs",
            columns=[ColumnSpec(name="id", type="INT"), ColumnSpec(name="body", type="TEXT")],
        )
    )
    first.add_index("docs", "body", "inverted", {"vocabulary": 5})

    second = _persistent_catalog(tmp_path)

    assert second.list_tables() == ["docs"]
    info = second.get_table("docs")
    assert info.schema.column_names() == ["id", "body"]
    assert info.schema.type_of("body") == "TEXT"
    assert len(info.indexes) == 1
    assert info.indexes[0].column == "body"
    assert info.indexes[0].index_type == "inverted"
    assert info.indexes[0].options == {"vocabulary": 5}


def test_catalog_snapshot_forgets_dropped_tables(tmp_path: Path) -> None:
    first = _persistent_catalog(tmp_path)
    first.create_table(Schema(table="docs", columns=[ColumnSpec(name="id", type="INT")]))
    first.create_table(Schema(table="img", columns=[ColumnSpec(name="id", type="INT")]))
    first.drop_table("docs")

    second = _persistent_catalog(tmp_path)

    assert second.list_tables() == ["img"]
    assert not second.has_table("docs")


def _session_over(base_dir: Path) -> tuple[Session, QueryExecutor, Catalog]:
    storage = FileStorageEngine(base_dir)
    executor = QueryExecutor(EngineIndexFactory(), storage)
    catalog = Catalog(buffer=LRUBufferManager(storage))
    session = Session(SqlParser(), QueryPlanner(), executor, catalog)
    return session, executor, catalog


def test_restart_recovers_text_search_without_repeating_ddl(tmp_path: Path) -> None:
    session, _executor, _catalog = _session_over(tmp_path)
    session.execute("CREATE TABLE docs (id INT, body TEXT)")
    session.execute("CREATE INDEX ON docs (body) USING inverted WITH (vocabulary = 5)")
    session.execute(
        'INSERT INTO docs (id, body) VALUES '
        '(1, "visual search database"), '
        '(2, "audio retrieval engine"), '
        '(3, "visual retrieval engine")'
    )

    # Un segundo executor sobre el mismo directorio simula el reinicio de la API
    restarted, executor, catalog = _session_over(tmp_path)
    rehydrate_executor(executor, catalog)

    result = restarted.execute('SELECT id FROM docs WHERE MATCH(body, "visual", 3)')

    assert {row[0] for row in result.rows} == {1, 3}
    assert result.index_type == "inverted"
    info = catalog.get_table("docs")
    assert info.indexes[0].options == {"vocabulary": 5}
