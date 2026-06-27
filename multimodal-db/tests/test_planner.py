import pytest

from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from query.plan_types import PlanOp
from indices.ports import (
    EqualityPredicate,
    KnnPredicate,
    RangePredicate,
    SpatialRangePredicate,
)


@pytest.fixture
def plan():
    parser = SqlParser()
    planner = QueryPlanner()
    return lambda sql: planner.plan(parser.parse(sql))


def test_create_table(plan):
    q = plan("CREATE TABLE img (id INT, path TEXT, feat VECTOR)")
    assert q.op is PlanOp.CREATE_TABLE
    assert q.columns == ["id", "path", "feat"]
    assert q.column_types == ["INT", "TEXT", "VECTOR"]


def test_drop_table(plan):
    q = plan("DROP TABLE img")
    assert q.op is PlanOp.DROP_TABLE
    assert q.table == "img"


def test_create_index(plan):
    q = plan("CREATE INDEX ON img (feat) USING rtree")
    assert q.op is PlanOp.CREATE_INDEX
    assert q.columns == ["feat"]
    assert q.index_type == "rtree"


def test_insert(plan):
    q = plan('INSERT INTO img (id, path) VALUES (1, "a.jpg"), (2, "b.jpg")')
    assert q.op is PlanOp.INSERT
    assert q.columns == ["id", "path"]
    assert q.rows == [(1, "a.jpg"), (2, "b.jpg")]


def test_delete_no_where(plan):
    q = plan("DELETE FROM img")
    assert q.op is PlanOp.DELETE
    assert q.predicate is None


def test_delete_equality(plan):
    q = plan("DELETE FROM img WHERE id = 5")
    assert isinstance(q.predicate, EqualityPredicate)
    assert q.predicate.value == 5


def test_delete_between(plan):
    q = plan("DELETE FROM img WHERE id BETWEEN 1 AND 9")
    assert isinstance(q.predicate, RangePredicate)
    assert (q.predicate.low, q.predicate.high) == (1, 9)


def test_select_all(plan):
    q = plan("SELECT * FROM img")
    assert q.op is PlanOp.SELECT
    assert q.predicate is None
    assert q.index_type is None


def test_select_equality_uses_hash(plan):
    q = plan("SELECT * FROM img WHERE id = 5")
    assert isinstance(q.predicate, EqualityPredicate)
    assert q.index_type == "hash"


def test_select_range_uses_bplus(plan):
    q = plan("SELECT * FROM img WHERE id BETWEEN 1 AND 9")
    assert isinstance(q.predicate, RangePredicate)
    assert q.index_type == "bplus"


def test_select_limit(plan):
    q = plan("SELECT id FROM img LIMIT 10")
    assert q.k == 10


@pytest.mark.parametrize(
    "op,low,high,inc_low,inc_high",
    [
        ("<", None, 5, True, False),
        ("<=", None, 5, True, True),
        (">", 5, None, False, True),
        (">=", 5, None, True, True),
    ],
)
def test_range_ops(plan, op, low, high, inc_low, inc_high):
    q = plan(f"SELECT * FROM img WHERE id {op} 5")
    assert isinstance(q.predicate, RangePredicate)
    assert q.predicate.low == low
    assert q.predicate.high == high
    assert q.predicate.include_low is inc_low
    assert q.predicate.include_high is inc_high


def test_select_knn_vector(plan):
    q = plan("SELECT * FROM img WHERE KNN(feat, [0.1, 0.2, 0.3], 5)")
    assert isinstance(q.predicate, KnnPredicate)
    assert q.predicate.query == [0.1, 0.2, 0.3]
    assert q.index_type == "knn"
    assert q.k == 5


def test_select_knn_file(plan):
    q = plan('SELECT * FROM img WHERE KNN(feat, "q.jpg", 8)')
    assert isinstance(q.predicate, KnnPredicate)
    assert q.predicate.query == "q.jpg"
    assert q.k == 8


def test_select_spatial_uses_rtree(plan):
    q = plan("SELECT * FROM img WHERE WITHIN(box, [0, 0], [10, 10])")
    assert isinstance(q.predicate, SpatialRangePredicate)
    assert q.predicate.min_corner == [0, 0]
    assert q.index_type == "rtree"


def test_unsupported_operator_raises(plan):
    with pytest.raises(ValueError):
        plan("DELETE FROM img WHERE id != 5")
