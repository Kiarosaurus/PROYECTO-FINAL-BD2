import pytest

from query.parser.sql_parser import SqlParser
from query.parser import ast as A


@pytest.fixture
def parser():
    return SqlParser()


def test_create_table(parser):
    stmt = parser.parse("CREATE TABLE img (id INT, path TEXT, feat VECTOR)")
    assert isinstance(stmt, A.CreateTable)
    assert stmt.table == "img"
    assert [c.name for c in stmt.columns] == ["id", "path", "feat"]
    assert [c.type for c in stmt.columns] == ["INT", "TEXT", "VECTOR"]


def test_drop_table(parser):
    stmt = parser.parse("DROP TABLE img")
    assert isinstance(stmt, A.DropTable)
    assert stmt.table == "img"


def test_create_index(parser):
    stmt = parser.parse("CREATE INDEX ON img (feat) USING rtree")
    assert isinstance(stmt, A.CreateIndex)
    assert (stmt.table, stmt.column, stmt.index_type) == ("img", "feat", "rtree")


def test_insert_multiple_rows(parser):
    stmt = parser.parse('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')
    assert isinstance(stmt, A.Insert)
    assert stmt.columns == ["id", "path"]
    assert stmt.rows == [(1, "a.jpg"), (2, "b.jpg")]


def test_insert_row_with_vector_literal(parser):
    stmt = parser.parse("INSERT INTO tracks (id, feat) VALUES (1, [0.5, -1.0, 2])")
    assert isinstance(stmt, A.Insert)
    assert stmt.rows == [(1, [0.5, -1.0, 2])]


def test_delete_no_where(parser):
    stmt = parser.parse("DELETE FROM img")
    assert isinstance(stmt, A.Delete)
    assert stmt.where is None


def test_delete_equality(parser):
    stmt = parser.parse("DELETE FROM img WHERE id = 5")
    assert isinstance(stmt.where, A.Comparison)
    assert (stmt.where.column, stmt.where.op, stmt.where.value) == ("id", "=", 5)


def test_select_all(parser):
    stmt = parser.parse("SELECT * FROM img")
    assert isinstance(stmt, A.Select)
    assert stmt.columns == ["*"]
    assert stmt.where is None
    assert stmt.limit is None


def test_select_projection_limit(parser):
    stmt = parser.parse("SELECT id, path FROM img LIMIT 10")
    assert stmt.columns == ["id", "path"]
    assert stmt.limit == 10


def test_select_between(parser):
    stmt = parser.parse("SELECT * FROM img WHERE id BETWEEN 1 AND 9")
    assert isinstance(stmt.where, A.Between)
    assert (stmt.where.low, stmt.where.high) == (1, 9)


@pytest.mark.parametrize("op", ["<", "<=", ">", ">=", "!="])
def test_comparison_ops(parser, op):
    stmt = parser.parse(f"SELECT * FROM img WHERE id {op} 5")
    assert isinstance(stmt.where, A.Comparison)
    assert stmt.where.op == op


def test_knn_vector(parser):
    stmt = parser.parse("SELECT * FROM img WHERE KNN(feat, [0.1, 0.2, 0.3], 5)")
    assert isinstance(stmt.where, A.KnnCondition)
    assert stmt.where.column == "feat"
    assert stmt.where.query == [0.1, 0.2, 0.3]
    assert stmt.where.k == 5


def test_knn_file(parser):
    stmt = parser.parse('SELECT * FROM img WHERE KNN(feat, "q.jpg", 8)')
    assert isinstance(stmt.where, A.KnnCondition)
    assert stmt.where.query == "q.jpg"
    assert stmt.where.k == 8


def test_spatial(parser):
    stmt = parser.parse("SELECT * FROM img WHERE WITHIN(box, [0, 0], [10, 10])")
    assert isinstance(stmt.where, A.SpatialCondition)
    assert stmt.where.min_corner == [0, 0]
    assert stmt.where.max_corner == [10, 10]


def test_match_with_k(parser):
    stmt = parser.parse('SELECT * FROM docs WHERE MATCH(body, "database systems", 5)')
    assert isinstance(stmt.where, A.MatchCondition)
    assert stmt.where.column == "body"
    assert stmt.where.terms == "database systems"
    assert stmt.where.k == 5


def test_match_without_k(parser):
    stmt = parser.parse('SELECT * FROM docs WHERE MATCH(body, "database")')
    assert isinstance(stmt.where, A.MatchCondition)
    assert stmt.where.terms == "database"
    assert stmt.where.k is None


def test_match_with_limit(parser):
    stmt = parser.parse('SELECT id FROM docs WHERE MATCH(body, "query engine", 3) LIMIT 10')
    assert isinstance(stmt.where, A.MatchCondition)
    assert stmt.where.k == 3
    assert stmt.limit == 10


@pytest.mark.parametrize("kw", ["match", "Match", "MATCH"])
def test_match_case_insensitive(parser, kw):
    stmt = parser.parse(f'SELECT * FROM docs WHERE {kw}(body, "text", 2)')
    assert isinstance(stmt.where, A.MatchCondition)
    assert stmt.where.k == 2


def test_string_value_unquoted(parser):
    stmt = parser.parse('DELETE FROM img WHERE path = "a.jpg"')
    assert stmt.where.value == "a.jpg"


def test_float_value(parser):
    stmt = parser.parse("SELECT * FROM img WHERE score >= 0.5")
    assert stmt.where.value == 0.5
