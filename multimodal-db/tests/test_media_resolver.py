from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.storage.file_engine import FileStorageEngine
from indices.ports import KnnPredicate
from multimedia.codebook.kmeans_codebook import KMeansCodebook
from multimedia.knn_index import MultimediaKNNIndex
from multimedia.ports.extractor import FeatureExtractor
from multimedia.resolver import CompositeMediaResolver, PipelineMediaResolver
from tests.mocks import MockMediaResolver


# Extractor determinista que deriva los descriptores del nombre del archivo
class _StubExtractor(FeatureExtractor):

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def extract(self, file_path: str) -> np.ndarray:
        seed = sum(Path(file_path).name.encode("utf-8"))
        rng = np.random.default_rng(seed)
        return rng.random((12, self._dim)).astype(np.float32)

    def feature_dim(self) -> int:
        return self._dim

    def supported_formats(self) -> list[str]:
        return [".png"]


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


def test_insert_vector_row_uses_first_column_as_key() -> None:
    index = MultimediaKNNIndex()

    inserted = index.insert([1.0, 0.0], {"id": 7, "feat": [1.0, 0.0]})
    result = index.search([1.0, 0.0], k=1)

    assert inserted.success is True
    assert [key for key, _score in result.records] == ["7"]


def test_resolver_persists_codebook_and_skips_refit_on_restart(tmp_path: Path, monkeypatch) -> None:
    media_dir = tmp_path / "uploads"
    media_dir.mkdir()
    for name in ["a.png", "b.png", "c.png"]:
        (media_dir / name).write_bytes(b"stub")
    storage = FileStorageEngine(tmp_path / "engine")

    first = PipelineMediaResolver(
        _StubExtractor(),
        KMeansCodebook(k=4, random_state=11),
        media_dir,
        storage=storage,
    )
    before = first.resolve("a.png")

    fresh = KMeansCodebook(k=4, random_state=11)

    def _fail_fit(descriptors) -> None:
        raise AssertionError("el codebook cargado no debe reentrenarse")

    monkeypatch.setattr(fresh, "fit", _fail_fit)
    second = PipelineMediaResolver(_StubExtractor(), fresh, media_dir, storage=storage)
    after = second.resolve("a.png")

    np.testing.assert_array_equal(before, after)


def test_resolver_without_storage_keeps_training_in_memory(tmp_path: Path) -> None:
    media_dir = tmp_path / "uploads"
    media_dir.mkdir()
    for name in ["a.png", "b.png"]:
        (media_dir / name).write_bytes(b"stub")

    resolver = PipelineMediaResolver(
        _StubExtractor(),
        KMeansCodebook(k=4, random_state=11),
        media_dir,
    )
    histogram = resolver.resolve("a.png")

    assert histogram.shape == (4,)


def test_composite_resolver_routes_by_extension() -> None:
    image = MockMediaResolver({"a.png": [1.0, 0.0]})
    audio = MockMediaResolver({"a.wav": [0.0, 1.0]}, formats=[".wav"])
    composite = CompositeMediaResolver([image, audio])

    assert composite.resolve("a.png").tolist() == [1.0, 0.0]
    assert composite.resolve("a.wav").tolist() == [0.0, 1.0]
    assert composite.supported_formats() == [".png", ".wav"]


def test_composite_resolver_rejects_unknown_extension() -> None:
    composite = CompositeMediaResolver([MockMediaResolver({})])

    with pytest.raises(ValueError):
        composite.resolve("a.txt")
