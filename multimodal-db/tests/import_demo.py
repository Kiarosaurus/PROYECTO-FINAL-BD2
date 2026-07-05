#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile
from pathlib import Path

from core.storage.file_engine import FileStorageEngine
from ingest.fma_loader import FMALoader
from ingest.importer import DatasetImporter
from ingest.spotify_loader import FEATURE_COLUMNS, SpotifyLoader
from query.executor import QueryExecutor
from query.index_factory import EngineIndexFactory
from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from service.catalog import Catalog
from service.session import Session

# Raíz local de la data del proyecto
DATA_ROOT = Path(
    os.environ.get("DATA_DIR", str(Path(__file__).resolve().parents[2] / "data" / "raw"))
)


def _build_session(media_dir: Path | None) -> Session:
    resolver = None
    if media_dir is not None:
        from multimedia.codebook.kmeans_codebook import KMeansCodebook
        from multimedia.extractors.mfcc_extractor import MFCCExtractor
        from multimedia.resolver import PipelineMediaResolver

        resolver = PipelineMediaResolver(MFCCExtractor(), KMeansCodebook(k=32), media_dir)
    storage_dir = tempfile.mkdtemp(prefix="mmdb_import_")
    executor = QueryExecutor(
        EngineIndexFactory(media_resolver=resolver),
        FileStorageEngine(storage_dir),
    )
    return Session(SqlParser(), QueryPlanner(), executor, Catalog())


def _print_report(report) -> None:
    print("tabla:", report.table)
    print("filas:", report.rows_inserted, "lotes:", report.batches)
    print("índices:", report.indexes)


def _spotify_demo(limit: int) -> None:
    features_csv = DATA_ROOT / "songs" / "audio_features_dataset.csv"
    lyrics_csv = DATA_ROOT / "lyrics" / "lyrics_dataset.csv"
    if not features_csv.is_file() or not lyrics_csv.is_file():
        print("spotify: no hay data local, corre tests/download_data.py primero")
        return
    session = _build_session(None)
    loader = SpotifyLoader(features_csv, lyrics_csv, DATA_ROOT / "images")
    report = DatasetImporter(session).run(loader, limit=limit)
    _print_report(report)

    result = session.execute('SELECT id, title FROM tracks WHERE MATCH(lyrics, "love night", 3)')
    print("match lyrics:", result.rows)

    with open(features_csv, newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    vector = "[" + ", ".join(f"{float(first[col]):.6f}" for col in FEATURE_COLUMNS) + "]"
    result = session.execute(f"SELECT * FROM tracks WHERE KNN(feat, {vector}, 3)")
    print("knn feat:", result.rows)


def _fma_demo(limit: int) -> None:
    tracks_csv = Path(os.environ.get("FMA_TRACKS_CSV", DATA_ROOT / "fma" / "tracks.csv"))
    audio_dir = Path(os.environ.get("FMA_AUDIO_DIR", DATA_ROOT / "fma" / "audio"))
    if not tracks_csv.is_file() or not audio_dir.is_dir():
        print("fma: no hay data local, se omite (ver FMA_TRACKS_CSV y FMA_AUDIO_DIR)")
        return
    session = _build_session(audio_dir)
    loader = FMALoader(tracks_csv, audio_dir)
    report = DatasetImporter(session).run(loader, limit=limit)
    _print_report(report)

    first = next(loader.rows(), None)
    if first is None:
        return
    result = session.execute(f'SELECT * FROM fma_tracks WHERE KNN(audio, "{first[5]}", 3)')
    print("knn audio:", result.rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["spotify", "fma"], default="spotify")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    if args.dataset == "spotify":
        _spotify_demo(args.limit)
    else:
        _fma_demo(args.limit)


if __name__ == "__main__":
    main()
