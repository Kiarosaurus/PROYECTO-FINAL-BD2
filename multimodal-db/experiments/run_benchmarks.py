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

# El módulo resource no existe en Windows
try:
    import resource
except ImportError:
    resource = None

VECTOR_DIM = 256
VOCABULARY_SIZE = 500
WORDS_PER_DOCUMENT = 20
TOP_K = 10
# Límite de vocabulario que usan las aplicaciones reales
DEFAULT_VOCABULARY = 8000
# Tope de descriptores SIFT usados para entrenar el codebook
DESCRIPTOR_SAMPLE = 100_000
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Orden fijo de columnas del CSV de resultados
CSV_COLUMNS = [
    "modality",
    "engine",
    "size",
    "build_s",
    "avg_query_ms",
    "disk_reads",
    "disk_writes",
    "throughput_qps",
    "recall_at_k",
    "overlap_at_k",
    "rss_mb",
    "pg_index_mb",
]

# Ruta por defecto del CSV de letras descargado de Drive
LYRICS_CSV = Path(__file__).resolve().parents[2] / "data" / "lyrics" / "lyrics_dataset.csv"

# Ruta por defecto de la carpeta de covers descargados de Drive
COVERS_DIR = Path(__file__).resolve().parents[2] / "data" / "images"


# Genera un vocabulario sintético con palabras que el preprocessor no descarta
def _vocabulary() -> list[str]:
    return [f"palabra{i:03d}" for i in range(VOCABULARY_SIZE)]


# Carga las letras del CSV y devuelve una lista de textos
def load_lyrics(csv_path: Path, limit: int) -> list[str]:
    csv.field_size_limit(1_000_000)
    documents = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lyrics = (row.get("lyrics") or "").strip()
            if lyrics:
                documents.append(lyrics)
            if len(documents) >= limit:
                break
    return documents


# Arma consultas con palabras tomadas de los propios documentos
def queries_from_documents(documents: list[str], count: int, rng: np.random.Generator) -> list[str]:
    queries = []
    for _ in range(count):
        words = documents[rng.integers(len(documents))].split()
        words = [w for w in words if len(w) >= 4] or words
        picked = rng.choice(words, size=min(2, len(words)), replace=False)
        queries.append(" ".join(picked))
    return queries


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


# Lista ordenada de imágenes de la carpeta de covers
def _cover_paths(covers_dir: Path) -> list[Path]:
    if not covers_dir.is_dir():
        return []
    return sorted(
        path for path in covers_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


# Convierte las primeras imágenes válidas en histogramas del codebook
def _compute_cover_histograms(paths: list[Path], target: int, seed: int) -> np.ndarray | None:
    from multimedia.codebook.kmeans_codebook import KMeansCodebook
    from multimedia.extractors.sift_extractor import SIFTExtractor

    extractor = SIFTExtractor()
    descriptors_list: list[np.ndarray] = []
    for path in paths:
        if len(descriptors_list) >= target:
            break
        try:
            descriptors = extractor.extract(str(path))
        except ValueError:
            continue
        if descriptors.shape[0] == 0:
            continue
        descriptors_list.append(descriptors)
    if not descriptors_list:
        return None
    stacked = np.vstack(descriptors_list)
    # El codebook necesita al menos un descriptor por cluster
    if stacked.shape[0] < VECTOR_DIM:
        return None
    sample = stacked
    if stacked.shape[0] > DESCRIPTOR_SAMPLE:
        picked = np.random.default_rng(seed).choice(
            stacked.shape[0], size=DESCRIPTOR_SAMPLE, replace=False
        )
        sample = stacked[picked]
    codebook = KMeansCodebook(k=VECTOR_DIM, random_state=seed)
    codebook.fit(sample)
    histograms = [codebook.quantize(descriptors) for descriptors in descriptors_list]
    # Con el IDF aprendido se recuantiza para aplicar los pesos nuevos
    codebook.compute_idf(histograms)
    histograms = [codebook.quantize(descriptors) for descriptors in descriptors_list]
    return np.vstack(histograms).astype(np.float32)


# Histogramas TF-IDF reales de los covers con caché en disco
# Si los covers no alcanzan se completa con muestreo con reemplazo
def load_cover_histograms(
    covers_dir: str | Path,
    count: int,
    rng: np.random.Generator,
    seed: int = 42,
) -> np.ndarray | None:
    covers_path = Path(covers_dir)
    paths = _cover_paths(covers_path)
    if not paths:
        return None
    target = min(count, len(paths))
    cache_file = covers_path.parent / f"covers_hist_{target}img_k{VECTOR_DIM}.npy"
    if cache_file.is_file():
        histograms = np.load(cache_file)
    else:
        histograms = _compute_cover_histograms(paths, target, seed)
        if histograms is None:
            return None
        np.save(cache_file, histograms)
    if histograms.shape[0] < count:
        extra = rng.integers(0, histograms.shape[0], count - histograms.shape[0])
        histograms = np.vstack([histograms, histograms[extra]])
    return histograms.astype(np.float32)


def _average_ms(samples: list[float]) -> float:
    return round(sum(samples) / max(len(samples), 1), 3)


# Consultas por segundo sobre el tiempo total del lote
def _throughput_qps(query_count: int, wall_s: float) -> float:
    if wall_s <= 0:
        return 0.0
    return round(query_count / wall_s, 2)


# Pico de memoria del proceso en MB
def process_peak_rss_mb() -> float:
    if resource is None:
        return 0.0
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# Calcula el top k real comparando cada consulta contra toda la colección
def exact_knn_ground_truth(vectors: np.ndarray, queries: np.ndarray, k: int = TOP_K) -> list[set[str]]:
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    truth: list[set[str]] = []
    for query in queries:
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            query_norm = 1.0
        similarities = (vectors @ query) / (norms * query_norm)
        top = np.argsort(similarities)[::-1][:k]
        truth.append({str(int(position)) for position in top})
    return truth


# Fracción promedio del top k real que cada motor logra devolver
def recall_at_k(retrieved: list[set[str]], truth: list[set[str]]) -> float:
    if not truth:
        return 0.0
    total = 0.0
    for found, expected in zip(retrieved, truth):
        total += len(found & expected) / max(len(expected), 1)
    return round(total / len(truth), 3)


# Parecido Jaccard promedio entre los top k de dos motores
def jaccard_overlap(first: list[set[str]], second: list[set[str]]) -> float:
    if not first:
        return 0.0
    total = 0.0
    for left, right in zip(first, second):
        union = left | right
        total += len(left & right) / len(union) if union else 1.0
    return round(total / len(first), 3)


# Tamaño en MB de un índice nativo de PostgreSQL
def _pg_index_size_mb(dsn: str, index_name: str) -> float:
    import psycopg2

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COALESCE(pg_relation_size(to_regclass(%s)), 0)", (index_name,))
        size_bytes = cur.fetchone()[0]
    return round(size_bytes / (1024 * 1024), 2)


# Con dos índices vectoriales presentes el planner puede elegir el otro
# Se borra el índice que no se va a medir para que la medición sea real
def _ensure_single_vector_index(dsn: str, kind: str) -> None:
    import psycopg2

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        if kind == "IVFFlat":
            cur.execute("DROP INDEX IF EXISTS compare.idx_media_hnsw")
        else:
            cur.execute("DROP INDEX IF EXISTS compare.idx_media_ivfflat")
        conn.commit()


# Vuelve a dejar solo el índice IVFFlat que define init.sql
def _restore_default_vector_index(dsn: str) -> None:
    import psycopg2

    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS compare.idx_media_hnsw")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_media_ivfflat "
            "ON compare.media USING ivfflat (feature_vec vector_cosine_ops) "
            "WITH (lists = 100)"
        )
        conn.commit()


def bench_own_text(
    documents: list[str],
    queries: list[str],
    workdir: Path,
    vocabulary: int | None = None,
) -> tuple[dict, list[set[str]]]:
    storage = FileStorageEngine(workdir)
    buffer = LRUBufferManager(storage, capacity=256)
    index = InvertedIndex(
        column="body",
        buffer=buffer,
        file_id="bench_text",
        vocabulary_limit=vocabulary,
    )
    records = [{"id": i, "body": body} for i, body in enumerate(documents)]
    start = time.perf_counter()
    index.build(records)
    build_s = round(time.perf_counter() - start, 3)
    latencies = []
    retrieved: list[set[str]] = []
    batch_start = time.perf_counter()
    for terms in queries:
        t0 = time.perf_counter()
        result = index.search(TextMatchPredicate(column="body", terms=terms, k=TOP_K))
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved.append({str(record["id"]) for record in result.records})
    wall_s = time.perf_counter() - batch_start
    stats = storage.stats()
    row = {
        "modality": "text",
        "engine": "own-inverted",
        "size": len(documents),
        "build_s": build_s,
        "avg_query_ms": _average_ms(latencies),
        "disk_reads": stats.disk_reads,
        "disk_writes": stats.disk_writes,
        "throughput_qps": _throughput_qps(len(queries), wall_s),
        "rss_mb": process_peak_rss_mb(),
    }
    return row, retrieved


def bench_own_knn(vectors: np.ndarray, queries: np.ndarray, workdir: Path) -> tuple[dict, list[set[str]]]:
    storage = FileStorageEngine(workdir)
    buffer = LRUBufferManager(storage, capacity=256)
    index = MultimediaKNNIndex(buffer=buffer, file_id="bench_knn")
    start = time.perf_counter()
    index.build([(str(i), vector) for i, vector in enumerate(vectors)])
    build_s = round(time.perf_counter() - start, 3)
    latencies = []
    retrieved: list[set[str]] = []
    batch_start = time.perf_counter()
    for query in queries:
        t0 = time.perf_counter()
        result = index.search(query, k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved.append({str(key) for key, _score in result.records})
    wall_s = time.perf_counter() - batch_start
    stats = storage.stats()
    row = {
        "modality": "vector",
        "engine": "own-knn",
        "size": len(vectors),
        "build_s": build_s,
        "avg_query_ms": _average_ms(latencies),
        "disk_reads": stats.disk_reads,
        "disk_writes": stats.disk_writes,
        "throughput_qps": _throughput_qps(len(queries), wall_s),
        "rss_mb": process_peak_rss_mb(),
    }
    return row, retrieved


def bench_pg_text(documents: list[str], queries: list[str], dsn: str) -> tuple[dict, list[set[str]]]:
    from comparison.postgres_gin import PostgresGINEngine

    engine = PostgresGINEngine(dsn)
    start = time.perf_counter()
    engine.load([{"lyrics": body} for body in documents])
    engine.build_native_index("GIN")
    build_s = round(time.perf_counter() - start, 3)
    latencies = []
    retrieved: list[set[str]] = []
    batch_start = time.perf_counter()
    for terms in queries:
        result = engine.query(terms)
        latencies.append(result.latency_ms)
        retrieved.append({str(item[0]) for item in result.records})
    wall_s = time.perf_counter() - batch_start
    row = {
        "modality": "text",
        "engine": "pg-gin",
        "size": len(documents),
        "build_s": build_s,
        "avg_query_ms": _average_ms(latencies),
        "disk_reads": 0,
        "disk_writes": 0,
        "throughput_qps": _throughput_qps(len(queries), wall_s),
        "pg_index_mb": _pg_index_size_mb(dsn, "compare.idx_documents_fts"),
    }
    return row, retrieved


def bench_pg_knn(
    vectors: np.ndarray,
    queries: np.ndarray,
    dsn: str,
    kind: str = "IVFFlat",
) -> tuple[dict, list[set[str]]]:
    from comparison.pgvector_ivf import PgVectorIVFEngine

    engine = PgVectorIVFEngine(dsn, modality="IMAGE")
    start = time.perf_counter()
    engine.load([{"path": f"v{i}", "vector": vector} for i, vector in enumerate(vectors)])
    engine.build_native_index(kind)
    build_s = round(time.perf_counter() - start, 3)
    latencies = []
    retrieved: list[set[str]] = []
    batch_start = time.perf_counter()
    for query in queries:
        result = engine.query(query)
        latencies.append(result.latency_ms)
        retrieved.append({str(item[0]) for item in result.records})
    wall_s = time.perf_counter() - batch_start
    if kind == "IVFFlat":
        engine_name = "pg-ivfflat"
        index_name = "compare.idx_media_ivfflat"
    else:
        engine_name = "pg-hnsw"
        index_name = "compare.idx_media_hnsw"
    row = {
        "modality": "vector",
        "engine": engine_name,
        "size": len(vectors),
        "build_s": build_s,
        "avg_query_ms": _average_ms(latencies),
        "disk_reads": 0,
        "disk_writes": 0,
        "throughput_qps": _throughput_qps(len(queries), wall_s),
        "pg_index_mb": _pg_index_size_mb(dsn, index_name),
    }
    return row, retrieved


# Corre todas las mediciones y deja CSV y plots en out_dir
def run_benchmarks(
    sizes: list[int],
    query_count: int = 10,
    out_dir: str | Path = "experiments/results/local",
    seed: int = 42,
    dsn: str | None = None,
    make_plots: bool = True,
    lyrics_csv: str | Path | None = None,
    covers_dir: str | Path | None = None,
    vocabulary: int | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    synthetic_words = _vocabulary()
    corpus: list[str] = []
    if lyrics_csv and Path(lyrics_csv).is_file():
        corpus = load_lyrics(Path(lyrics_csv), limit=max(sizes))
    for size in sizes:
        if corpus:
            documents = corpus[:size]
            # Si el corpus real no alcanza se completa con muestreo con reemplazo
            if len(documents) < size:
                extra = rng.integers(0, len(corpus), size - len(documents))
                documents = documents + [corpus[int(i)] for i in extra]
            text_queries = queries_from_documents(documents, query_count, rng)
        else:
            documents = make_documents(size, rng)
            text_queries = [
                " ".join(rng.choice(synthetic_words, size=2))
                for _ in range(query_count)
            ]
        vectors = None
        if covers_dir is not None:
            vectors = load_cover_histograms(covers_dir, size, rng, seed=seed)
            if vectors is None:
                print(f"sin imágenes válidas en {covers_dir}, se usan vectores sintéticos")
                covers_dir = None
        if vectors is None:
            vectors = make_vectors(size, rng)
        vector_queries = vectors[: min(query_count, len(vectors))]
        truth = exact_knn_ground_truth(vectors, vector_queries, TOP_K)
        with tempfile.TemporaryDirectory() as workdir:
            text_row, own_text_top = bench_own_text(
                documents, text_queries, Path(workdir), vocabulary=vocabulary
            )
        rows.append(text_row)
        with tempfile.TemporaryDirectory() as workdir:
            knn_row, own_knn_top = bench_own_knn(vectors, vector_queries, Path(workdir))
        knn_row["recall_at_k"] = recall_at_k(own_knn_top, truth)
        rows.append(knn_row)
        if dsn:
            gin_row, gin_top = bench_pg_text(documents, text_queries, dsn)
            text_row["overlap_at_k"] = jaccard_overlap(own_text_top, gin_top)
            rows.append(gin_row)
            for kind in ("IVFFlat", "HNSW"):
                _ensure_single_vector_index(dsn, kind)
                pg_row, pg_top = bench_pg_knn(vectors, vector_queries, dsn, kind)
                pg_row["recall_at_k"] = recall_at_k(pg_top, truth)
                rows.append(pg_row)
            _restore_default_vector_index(dsn)
    _write_csv(rows, out_path / "results.csv")
    if make_plots:
        _write_plots(rows, out_path)
    return rows


def _write_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


# Un plot comparativo de latencia por modalidad y otro de recall
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
    subset = [
        row
        for row in rows
        if row["modality"] == "vector" and row.get("recall_at_k", "") != ""
    ]
    if not subset:
        return
    engines = sorted({row["engine"] for row in subset})
    figure, axis = plt.subplots(figsize=(7, 4))
    for engine in engines:
        points = sorted(
            (row["size"], row["recall_at_k"])
            for row in subset
            if row["engine"] == engine
        )
        axis.plot(
            [size for size, _recall in points],
            [recall for _size, recall in points],
            marker="o",
            label=engine,
        )
    axis.set_xlabel("cantidad de registros")
    axis.set_ylabel(f"recall@{TOP_K}")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Recall contra el scan lineal exacto")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_path / "recall_vector.png", dpi=120)
    plt.close(figure)


# Formatea una celda que puede venir vacía
def _fmt_metric(value: object) -> str:
    if value in ("", None):
        return "     -"
    return f"{float(value):6.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark del engine propio contra PostgreSQL")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000])
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--out", default="experiments/results/local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lyrics-csv", default=str(LYRICS_CSV))
    parser.add_argument(
        "--covers-dir",
        default=str(COVERS_DIR),
        help="carpeta con covers reales para los histogramas del codebook",
    )
    parser.add_argument(
        "--vocabulary",
        type=int,
        default=DEFAULT_VOCABULARY,
        help="límite top-k del vocabulario del índice de texto, 0 lo desactiva",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="usa corpus y vectores sintéticos en vez de la data real",
    )
    args = parser.parse_args()
    dsn = os.environ.get("POSTGRES_DSN")
    lyrics_csv = None if args.synthetic else args.lyrics_csv
    if lyrics_csv and not Path(lyrics_csv).is_file():
        print(f"no se encontró {lyrics_csv}, se usa corpus sintético")
        lyrics_csv = None
    covers_dir = None if args.synthetic else args.covers_dir
    if covers_dir and not Path(covers_dir).is_dir():
        print(f"no se encontró {covers_dir}, se usan vectores sintéticos")
        covers_dir = None
    vocabulary = args.vocabulary if args.vocabulary > 0 else None
    rows = run_benchmarks(
        sizes=args.sizes,
        query_count=args.queries,
        out_dir=args.out,
        seed=args.seed,
        dsn=dsn,
        lyrics_csv=lyrics_csv,
        covers_dir=covers_dir,
        vocabulary=vocabulary,
    )
    for row in rows:
        print(
            f"{row['modality']:6} | {row['engine']:12} | n={row['size']:7} | "
            f"build={row['build_s']:8.3f}s | query={row['avg_query_ms']:8.3f}ms | "
            f"qps={row['throughput_qps']:9.2f} | "
            f"recall={_fmt_metric(row.get('recall_at_k'))} | "
            f"overlap={_fmt_metric(row.get('overlap_at_k'))} | "
            f"rss={_fmt_metric(row.get('rss_mb'))}MB | "
            f"pg_idx={_fmt_metric(row.get('pg_index_mb'))}MB"
        )
    if not dsn:
        print("POSTGRES_DSN no definido, se midieron solo los índices propios")


if __name__ == "__main__":
    main()
