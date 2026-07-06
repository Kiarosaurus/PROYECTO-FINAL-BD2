from __future__ import annotations

from pathlib import Path

from experiments.run_benchmarks import (
    exact_knn_ground_truth,
    jaccard_overlap,
    load_cover_histograms,
    make_documents,
    make_vectors,
    recall_at_k,
    run_benchmarks,
)

import numpy as np


def test_benchmark_generates_csv_and_rows_without_postgres(tmp_path: Path) -> None:
    rows = run_benchmarks(
        sizes=[30],
        query_count=3,
        out_dir=tmp_path,
        seed=7,
        dsn=None,
        make_plots=False,
    )

    engines = {row["engine"] for row in rows}
    assert engines == {"own-inverted", "own-knn"}
    assert all(row["size"] == 30 for row in rows)
    assert all(row["avg_query_ms"] >= 0 for row in rows)
    assert all(row["throughput_qps"] > 0 for row in rows)
    knn_row = next(row for row in rows if row["engine"] == "own-knn")
    assert 0.0 <= knn_row["recall_at_k"] <= 1.0
    assert all(row["rss_mb"] >= 0 for row in rows)
    assert (tmp_path / "results.csv").exists()


def test_recall_at_k_with_exact_ground_truth_minimal() -> None:
    vectors = np.eye(3, dtype=np.float32)
    truth = exact_knn_ground_truth(vectors, vectors, k=1)

    assert truth == [{"0"}, {"1"}, {"2"}]
    assert recall_at_k(truth, truth) == 1.0
    assert recall_at_k([{"0"}, {"1"}, {"9"}], truth) == round(2 / 3, 3)


def test_small_corpus_is_amplified_by_resampling(tmp_path: Path) -> None:
    lyrics_csv = tmp_path / "lyrics.csv"
    lyrics_csv.write_text(
        "track_id,lyrics\na,uno dos tres cuatro\nb,cinco seis siete ocho\n",
        encoding="utf-8",
    )
    rows = run_benchmarks(
        sizes=[8],
        query_count=2,
        out_dir=tmp_path,
        seed=5,
        dsn=None,
        make_plots=False,
        lyrics_csv=lyrics_csv,
    )

    assert all(row["size"] == 8 for row in rows)


def test_jaccard_overlap_minimal() -> None:
    assert jaccard_overlap([{"a", "b"}], [{"a", "b"}]) == 1.0
    assert jaccard_overlap([{"a", "b"}], [{"a", "c"}]) == round(1 / 3, 3)
    assert jaccard_overlap([], []) == 0.0


def test_benchmark_own_text_reports_real_disk_io(tmp_path: Path) -> None:
    rows = run_benchmarks(
        sizes=[20],
        query_count=2,
        out_dir=tmp_path,
        seed=1,
        dsn=None,
        make_plots=False,
    )

    text_row = next(row for row in rows if row["engine"] == "own-inverted")
    assert text_row["disk_writes"] > 0


def test_benchmark_own_knn_reports_real_disk_io(tmp_path: Path) -> None:
    rows = run_benchmarks(
        sizes=[20],
        query_count=2,
        out_dir=tmp_path,
        seed=1,
        dsn=None,
        make_plots=False,
    )

    knn_row = next(row for row in rows if row["engine"] == "own-knn")
    assert knn_row["disk_writes"] > 0


def test_dataset_generators_are_deterministic() -> None:
    first = make_documents(5, np.random.default_rng(3))
    second = make_documents(5, np.random.default_rng(3))
    assert first == second

    vectors = make_vectors(4, np.random.default_rng(3))
    assert vectors.shape == (4, 256)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(norms <= 1.001)


# Genera covers de ruido con keypoints suficientes para SIFT
def _noise_covers(folder: Path, count: int = 5) -> None:
    import cv2

    folder.mkdir(parents=True)
    for seed in range(count):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 255, size=(128, 128), dtype=np.uint8)
        cv2.imwrite(str(folder / f"noise_{seed}.png"), image)


def test_cover_histograms_shape_norms_and_amplification(tmp_path: Path) -> None:
    covers = tmp_path / "covers"
    _noise_covers(covers, count=5)

    histograms = load_cover_histograms(covers, 8, np.random.default_rng(0), seed=11)

    assert histograms.shape == (8, 256)
    norms = np.linalg.norm(histograms, axis=1)
    assert np.all(norms <= 1.001)
    # Las filas amplificadas son copias de los histogramas base
    base = histograms[:5]
    for row in histograms[5:]:
        assert any(np.array_equal(row, original) for original in base)


def test_cover_histograms_are_deterministic(tmp_path: Path) -> None:
    covers = tmp_path / "covers"
    _noise_covers(covers, count=4)

    first = load_cover_histograms(covers, 4, np.random.default_rng(0), seed=11)
    cache = tmp_path / "covers_hist_4img_k256.npy"
    assert cache.is_file()
    cache.unlink()
    second = load_cover_histograms(covers, 4, np.random.default_rng(0), seed=11)

    assert np.array_equal(first, second)


def test_benchmark_falls_back_to_synthetic_without_covers(tmp_path: Path) -> None:
    rows = run_benchmarks(
        sizes=[20],
        query_count=2,
        out_dir=tmp_path,
        seed=3,
        dsn=None,
        make_plots=False,
        covers_dir=tmp_path / "no_covers",
    )

    engines = {row["engine"] for row in rows}
    assert engines == {"own-inverted", "own-knn"}
    assert (tmp_path / "results.csv").exists()


def test_benchmark_own_text_with_vocabulary_limit(tmp_path: Path) -> None:
    rows = run_benchmarks(
        sizes=[20],
        query_count=2,
        out_dir=tmp_path,
        seed=3,
        dsn=None,
        make_plots=False,
        vocabulary=50,
    )

    text_row = next(row for row in rows if row["engine"] == "own-inverted")
    assert text_row["size"] == 20
    assert text_row["avg_query_ms"] >= 0
    assert text_row["throughput_qps"] > 0
