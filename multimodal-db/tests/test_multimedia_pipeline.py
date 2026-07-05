from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from multimedia.codebook.kmeans_codebook import KMeansCodebook
from multimedia.extractors.sift_extractor import SIFTExtractor
from multimedia.knn_index import MultimediaKNNIndex
from multimedia.pipeline import MultimediaPipeline
from tests.mocks import MockStorageEngine


@pytest.fixture
def image_paths(tmp_path: Path) -> list[str]:
    rng = np.random.default_rng(5)
    paths = []
    for i in range(3):
        image = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
        path = tmp_path / f"img{i}.png"
        cv2.imwrite(str(path), image)
        paths.append(str(path))
    return paths


def _pipeline() -> MultimediaPipeline:
    return MultimediaPipeline(SIFTExtractor(), KMeansCodebook(k=8), MultimediaKNNIndex())


def test_pipeline_builds_index_from_real_images(image_paths) -> None:
    pipeline = _pipeline()

    result = pipeline.build_from_files(image_paths)

    assert result.success
    assert result.affected == 3


def test_pipeline_query_image_finds_itself_first(image_paths) -> None:
    pipeline = _pipeline()
    pipeline.build_from_files(image_paths)

    top = pipeline.search_file(image_paths[0], k=3)

    assert top.records[0][0] == "img0.png"
    assert top.records[0][1] > top.records[-1][1]


def test_pipeline_persists_codebook_and_index_via_storage_port(image_paths) -> None:
    pipeline = _pipeline()
    pipeline.build_from_files(image_paths)
    storage = MockStorageEngine()

    pipeline.save(storage)

    assert storage.read_page("codebook", 0) != b""
    assert storage.read_page("knn_index", 0) != b""


def test_pipeline_fails_cleanly_without_descriptors(tmp_path: Path) -> None:
    flat = np.zeros((16, 16), dtype=np.uint8)
    path = tmp_path / "flat.png"
    cv2.imwrite(str(path), flat)
    pipeline = _pipeline()

    result = pipeline.build_from_files([str(path)])

    assert not result.success
