from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest

from multimedia.codebook.kmeans_codebook import KMeansCodebook
from multimedia.extractors.audio_extractor import AudioExtractor
from multimedia.extractors.sift_extractor import SIFTExtractor
from tests.mocks import MockCodebook, MockFeatureExtractor, MockStorageEngine


@pytest.fixture
def sample_image_path(tmp_path):
    # Crea una imagen chica con ruido para que SIFT encuentre keypoints
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def sample_audio_extractor(tmp_path):
    # CSV chico con una sola fila de features de audio
    path = tmp_path / "audio.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "track_id", "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo",
        ])
        writer.writerow(["track1", 0.5, 0.6, 1, -5.0, 1, 0.05, 0.1, 0.0, 0.2, 0.7, 120.0])
    return AudioExtractor(str(path))


# Cada caso entrega su propio extractor junto a un archivo de entrada válido
@pytest.fixture(params=["sift", "audio", "mock"])
def extractor_case(request, sample_image_path, sample_audio_extractor):
    if request.param == "sift":
        return SIFTExtractor(), sample_image_path
    if request.param == "audio":
        return sample_audio_extractor, "track1.wav"
    return MockFeatureExtractor(), sample_image_path


def test_feature_extractor_contract(extractor_case):
    extractor, file_path = extractor_case

    dim = extractor.feature_dim()
    assert isinstance(dim, int)
    assert dim > 0

    formats = extractor.supported_formats()
    assert isinstance(formats, list)

    vectors = extractor.extract(file_path)
    assert isinstance(vectors, np.ndarray)
    if vectors.shape[0] > 0:
        assert vectors.shape[1] == dim


@pytest.fixture(params=["kmeans", "mock"])
def codebook_case(request):
    if request.param == "kmeans":
        return KMeansCodebook(k=4)
    return MockCodebook()


def test_codebook_contract(codebook_case):
    rng = np.random.default_rng(0)
    descriptors = rng.random((50, 8)).astype(np.float32)

    codebook_case.fit(descriptors)
    labels = codebook_case.quantize(descriptors)
    assert isinstance(labels, np.ndarray)

    # save no debe reventar contra cualquier StorageEngine, aunque sea no-op
    codebook_case.save(MockStorageEngine())


def test_kmeans_codebook_save_and_load_round_trip():
    rng = np.random.default_rng(1)
    descriptors = rng.random((30, 8)).astype(np.float32)

    codebook = KMeansCodebook(k=4)
    codebook.fit(descriptors)
    before = codebook.quantize(descriptors)

    storage = MockStorageEngine()
    codebook.save(storage)

    restored = KMeansCodebook(k=4)
    restored.load(storage)
    after = restored.quantize(descriptors)

    np.testing.assert_array_equal(before, after)
