from __future__ import annotations

import csv

import cv2
import numpy as np
import pytest
import soundfile as sf

from multimedia.codebook.kmeans_codebook import KMeansCodebook
from multimedia.extractors.audio_extractor import AudioExtractor
from multimedia.extractors.mfcc_extractor import MFCCExtractor
from multimedia.extractors.sift_extractor import SIFTExtractor
from tests.mocks import MockCodebook, MockFeatureExtractor, MockStorageEngine


# Genera un wav con una señal seno para las pruebas
def _write_sine_wav(path, freq: float, duration: float, sr: int = 22050) -> str:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(str(path), signal, sr)
    return str(path)


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


@pytest.fixture
def sample_wav_path(tmp_path):
    return _write_sine_wav(tmp_path / "sample.wav", freq=440.0, duration=2.0)


# Cada caso entrega su propio extractor junto a un archivo de entrada válido
@pytest.fixture(params=["sift", "audio", "mfcc", "mock"])
def extractor_case(request, sample_image_path, sample_audio_extractor, sample_wav_path):
    if request.param == "sift":
        return SIFTExtractor(), sample_image_path
    if request.param == "audio":
        return sample_audio_extractor, "track1.wav"
    if request.param == "mfcc":
        return MFCCExtractor(), sample_wav_path
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


def test_mfcc_long_audio_yields_multiple_windows(sample_wav_path):
    extractor = MFCCExtractor()
    vectors = extractor.extract(sample_wav_path)
    assert vectors.dtype == np.float32
    assert vectors.shape[0] > 1
    assert vectors.shape[1] == extractor.feature_dim()


def test_mfcc_short_audio_single_window(tmp_path):
    # Audio de duración menor a una ventana completa
    path = _write_sine_wav(tmp_path / "short.wav", freq=440.0, duration=0.05)
    extractor = MFCCExtractor()
    vectors = extractor.extract(path)
    assert vectors.shape == (1, extractor.feature_dim())
    assert np.isfinite(vectors).all()


def test_mfcc_dim_stable_across_files(tmp_path):
    # Archivos con distinta frecuencia y duración comparten dimensión
    path_a = _write_sine_wav(tmp_path / "a.wav", freq=220.0, duration=1.0)
    path_b = _write_sine_wav(tmp_path / "b.wav", freq=880.0, duration=3.0)
    extractor = MFCCExtractor()
    vectors_a = extractor.extract(path_a)
    vectors_b = extractor.extract(path_b)
    assert vectors_a.shape[1] == vectors_b.shape[1] == extractor.feature_dim()
    assert vectors_b.shape[0] > vectors_a.shape[0]


def test_mfcc_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"esto no es audio")
    extractor = MFCCExtractor()
    with pytest.raises(ValueError, match="No se pudo leer el audio"):
        extractor.extract(str(path))


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
