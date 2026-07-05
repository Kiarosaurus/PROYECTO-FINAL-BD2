from __future__ import annotations

from pathlib import Path

from experiments.run_benchmarks import make_documents, make_vectors, run_benchmarks

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
    assert (tmp_path / "results.csv").exists()


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


def test_dataset_generators_are_deterministic() -> None:
    first = make_documents(5, np.random.default_rng(3))
    second = make_documents(5, np.random.default_rng(3))
    assert first == second

    vectors = make_vectors(4, np.random.default_rng(3))
    assert vectors.shape == (4, 256)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(norms <= 1.001)
