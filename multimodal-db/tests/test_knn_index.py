from __future__ import annotations

from pathlib import Path

import numpy as np

from core.buffer.lru_buffer import LRUBufferManager
from core.storage.file_engine import FileStorageEngine
from indices.ports import KnnPredicate
from multimedia.knn_index import MultimediaKNNIndex
from tests.mocks import MockStorageEngine


def test_knn_index_save_and_load_round_trip_without_pickle() -> None:
    index = MultimediaKNNIndex()
    index.build(
        [
            ("a", [1.0, 0.0, 0.0]),
            ("b", [0.0, 1.0, 0.0]),
            ("c", [1.0, 1.0, 0.0]),
        ]
    )
    storage = MockStorageEngine()
    index.save(storage)

    restored = MultimediaKNNIndex()
    restored.load(storage)
    result = restored.search(KnnPredicate(column="vec", query=[1.0, 0.0, 0.0], k=2))

    assert [key for key, _score in result.records] == ["a", "c"]
    page = storage.read_page("knn_index", 0)
    assert page.startswith(b"{")


def test_knn_index_with_buffer_persists_and_reloads_without_build(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    records = [(f"v{i}", rng.random(16).astype(np.float32)) for i in range(100)]
    storage = FileStorageEngine(tmp_path)
    buffer = LRUBufferManager(storage)
    index = MultimediaKNNIndex(buffer=buffer, file_id="knn_media_feat")
    built = index.build(records)
    query = records[3][1]
    expected = index.search(KnnPredicate(column="feat", query=query, k=5))

    restored = MultimediaKNNIndex(buffer=buffer, file_id="knn_media_feat")
    result = restored.search(KnnPredicate(column="feat", query=query, k=5))

    assert built.success
    assert storage.stats().disk_writes > 0
    # El snapshot ocupa varias páginas en lugar de un solo bloque
    assert storage.read_page("knn_media_feat", 2) != b""
    assert [key for key, _score in result.records] == [key for key, _score in expected.records]
    assert result.records[0][0] == "v3"


def test_knn_index_candidate_filter_falls_back_to_full_scan() -> None:
    index = MultimediaKNNIndex(candidate_ratio=1.0)
    index.build(
        [
            ("a", [1.0, 0.0]),
            ("b", [0.0, 1.0]),
        ]
    )

    result = index.search(np.array([1.0, 0.0], dtype=np.float32), k=2)

    assert [key for key, _score in result.records] == ["a", "b"]
