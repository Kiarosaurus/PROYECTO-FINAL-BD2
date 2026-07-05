from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from core.buffer.lru_buffer import LRUBufferManager
from core.storage.file_engine import FileStorageEngine
from indices.inverted.text_index import InvertedIndex
from indices.ports import TextMatchPredicate
from multimedia.knn_index import MultimediaKNNIndex

VECTOR_DIM = 256
VOCABULARY_SIZE = 500
WORDS_PER_DOCUMENT = 20


# Genera un vocabulario sintético con palabras que el preprocessor no descarta
def _vocabulary() -> list[str]:
    return [f"palabra{i:03d}" for i in range(VOCABULARY_SIZE)]


# Genera documentos con palabras frecuentes y palabras raras
def make_documents(count: int, rng: np.random.Generator) -> list[str]:
    vocabulary = _vocabulary()
    weights = np.array([1.0 / (rank + 1) for rank in range(len(vocabulary))])
    weights = weights / weights.sum()
    documents = []
    for _ in range(count):
        words = rng.choice(vocabulary, size=WORDS_PER_DOCUMENT, p=weights)
        documents.append(" ".join(words))
    return documents


# Genera histogramas dispersos parecidos a los del codebook
def make_vectors(count: int, rng: np.random.Generator) -> np.ndarray:
    vectors = rng.random((count, VECTOR_DIM)).astype(np.float32)
    mask = rng.random((count, VECTOR_DIM)) < 0.1
    vectors = vectors * mask
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def _average_ms(samples: list[float]) -> float:
    return round(sum(samples) / max(len(samples), 1), 3)


def bench_own_text(documents: list[str], queries: list[str], workdir: Path) -> dict:
    storage = FileStorageEngine(workdir)
    buffer = LRUBufferManager(storage, capacity=256)
    index = InvertedIndex(column="body", buffer=buffer, file_id="bench_text")
    records = [{"id": i, "body": body} for i, body in enumerate(documents)]
    start = time.perf_counter()
    index.build(records)
    build_s = round(time.perf_counter() - start, 3)
    latencies = []
    for terms in queries:
        t0 = time.perf_counter()
        index.search(TextMatchPredicate(column="body", terms=terms, k=10))
        latencies.append((time.perf_counter() - t0) * 1000)
    stats = storage.stats()
    return {
        "modality": "text",
        "engine": "own-inverted",
        "size": len(documents),
        "build_s": build_s,
        "avg_query_ms": _average_ms(latencies),
        "disk_reads": stats.disk_reads,
        "disk_writes": stats.disk_writes,
    }


def bench_own_knn(vectors: np.ndarray, queries: np.ndarray) -> dict:
    index = MultimediaKNNIndex()
    start = time.perf_counter()
    index.build([(str(i), vector) for i, vector in enumerate(vectors)])
    build_s = round(time.perf_counter() - start, 3)
    latencies = []
    for query in queries:
        t0 = time.perf_counter()
        index.search(query, k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "modality": "vector",
        "engine": "own-knn",
        "size": len(vectors),
        "build_s": build_s,
        "avg_query_ms": _average_ms(latencies),
        "disk_reads": 0,
        "disk_writes": 0,
    }


def bench_pg_text(documents: list[str], queries: list[str], dsn: str) -> dict:
    from comparison.postgres_gin import PostgresGINEngine

    engine = PostgresGINEngine(dsn)
    start = time.perf_counter()
    engine.load([{"lyrics": body} for body in documents])
    engine.build_native_index("GIN")
    build_s = round(time.perf_counter() - start, 3)
    latencies = [engine.query(terms).latency_ms for terms in queries]
    return {
        "modality": "text",
        "engine": "pg-gin",
        "size": len(documents),
        "build_s": build_s,
        "avg_query_ms": _average_ms(latencies),
        "disk_reads": 0,
        "disk_writes": 0,
    }


def bench_pg_knn(vectors: np.ndarray, queries: np.ndarray, dsn: str) -> dict:
    from comparison.pgvector_ivf import PgVectorIVFEngine

    engine = PgVectorIVFEngine(dsn, modality="IMAGE")
    start = time.perf_counter()
    engine.load([{"path": f"v{i}", "vector": vector} for i, vector in enumerate(vectors)])
    engine.build_native_index("IVFFlat")
    build_s = round(time.perf_counter() - start, 3)
    latencies = [engine.query(query).latency_ms for query in queries]
    return {
        "modality": "vector",
        "engine": "pg-ivfflat",
        "size": len(vectors),
        "build_s": build_s,
        "avg_query_ms": _average_ms(latencies),
        "disk_reads": 0,
        "disk_writes": 0,
    }


# Corre todas las mediciones y deja CSV y plots en out_dir
def run_benchmarks(
    sizes: list[int],
    query_count: int = 10,
    out_dir: str | Path = "experiments/results",
    seed: int = 42,
    dsn: str | None = None,
    make_plots: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    vocabulary = _vocabulary()
    for size in sizes:
        documents = make_documents(size, rng)
        text_queries = [
            " ".join(rng.choice(vocabulary, size=2))
            for _ in range(query_count)
        ]
        vectors = make_vectors(size, rng)
        vector_queries = vectors[: min(query_count, len(vectors))]
        with tempfile.TemporaryDirectory() as workdir:
            rows.append(bench_own_text(documents, text_queries, Path(workdir)))
        rows.append(bench_own_knn(vectors, vector_queries))
        if dsn:
            rows.append(bench_pg_text(documents, text_queries, dsn))
            rows.append(bench_pg_knn(vectors, vector_queries, dsn))
    _write_csv(rows, out_path / "results.csv")
    if make_plots:
        _write_plots(rows, out_path)
    return rows


def _write_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Un plot comparativo de latencia por modalidad
def _write_plots(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for modality in ("text", "vector"):
        subset = [row for row in rows if row["modality"] == modality]
        if not subset:
            continue
        engines = sorted({row["engine"] for row in subset})
        figure, axis = plt.subplots(figsize=(7, 4))
        for engine in engines:
            points = sorted(
                (row["size"], row["avg_query_ms"])
                for row in subset
                if row["engine"] == engine
            )
            axis.plot(
                [size for size, _latency in points],
                [latency for _size, latency in points],
                marker="o",
                label=engine,
            )
        axis.set_xlabel("cantidad de registros")
        axis.set_ylabel("latencia promedio (ms)")
        axis.set_title(f"Latencia de consulta ({modality})")
        axis.legend()
        figure.tight_layout()
        figure.savefig(out_path / f"latency_{modality}.png", dpi=120)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark del engine propio contra PostgreSQL")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--out", default="experiments/results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dsn = os.environ.get("POSTGRES_DSN")
    rows = run_benchmarks(
        sizes=args.sizes,
        query_count=args.queries,
        out_dir=args.out,
        seed=args.seed,
        dsn=dsn,
    )
    for row in rows:
        print(
            f"{row['modality']:6} | {row['engine']:12} | n={row['size']:7} | "
            f"build={row['build_s']:8.3f}s | query={row['avg_query_ms']:8.3f}ms"
        )
    if not dsn:
        print("POSTGRES_DSN no definido, se midieron solo los índices propios")


if __name__ == "__main__":
    main()
