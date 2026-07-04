from __future__ import annotations

import numpy as np

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
