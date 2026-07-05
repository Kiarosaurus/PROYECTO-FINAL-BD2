from __future__ import annotations

from indices.ports import KnnPredicate
from multimedia.knn_index import MultimediaKNNIndex
from tests.mocks import MockMediaResolver


def test_search_by_file_without_resolver_fails_cleanly() -> None:
    index = MultimediaKNNIndex()
    index.build([("a", [1.0, 0.0])])

    result = index.search(KnnPredicate(column="feat", query="query.png", k=1))

    assert result.success is False
    assert "resolver" in (result.message or "")


def test_search_by_file_with_resolver_ranks_results() -> None:
    resolver = MockMediaResolver({"query.png": [1.0, 0.0, 0.0]})
    index = MultimediaKNNIndex(resolver=resolver)
    index.build(
        [
            ("a", [1.0, 0.0, 0.0]),
            ("b", [0.0, 1.0, 0.0]),
            ("c", [0.9, 0.1, 0.0]),
        ]
    )

    result = index.search(KnnPredicate(column="feat", query="query.png", k=2))

    assert result.success is True
    assert [key for key, _score in result.records] == ["a", "c"]


def test_insert_row_resolves_file_reference() -> None:
    resolver = MockMediaResolver({"a.png": [1.0, 0.0], "q.png": [1.0, 0.0]})
    index = MultimediaKNNIndex(resolver=resolver)

    inserted = index.insert("a.png", {"id": 1, "feat": "a.png"})
    result = index.search(KnnPredicate(column="feat", query="q.png", k=1))

    assert inserted.success is True
    assert [key for key, _score in result.records] == ["a.png"]


def test_build_resolves_file_references() -> None:
    resolver = MockMediaResolver({"a.png": [1.0, 0.0], "b.png": [0.0, 1.0]})
    index = MultimediaKNNIndex(resolver=resolver)

    built = index.build([("a.png", "a.png"), ("b.png", "b.png")])
    result = index.search(resolver.resolve("a.png"), k=1)

    assert built.success is True
    assert [key for key, _score in result.records] == ["a.png"]


def test_search_missing_file_reports_failure() -> None:
    resolver = MockMediaResolver({})
    index = MultimediaKNNIndex(resolver=resolver)
    index.build([("a", [1.0, 0.0])])

    result = index.search(KnnPredicate(column="feat", query="ghost.png", k=1))

    assert result.success is False
    assert "ghost.png" in (result.message or "")


def test_insert_unresolvable_record_fails_cleanly() -> None:
    index = MultimediaKNNIndex()

    result = index.insert("x.png", {"id": 1, "feat": "x.png"})

    assert result.success is False
    assert result.affected == 0
