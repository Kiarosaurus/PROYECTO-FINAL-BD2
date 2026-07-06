from __future__ import annotations

import csv
from pathlib import Path

from typing import Iterator

from core.storage.file_engine import FileStorageEngine
from ingest.fma_loader import FMALoader
from ingest.importer import DatasetImporter
from ingest.ports import DatasetLoader
from ingest.spotify_loader import SpotifyLoader
from query.executor import QueryExecutor
from query.index_factory import EngineIndexFactory
from query.parser.sql_parser import SqlParser
from query.planner import QueryPlanner
from service.catalog import Catalog
from service.dto import ColumnSpec, IndexInfo
from service.session import Session
from tests.test_integration_sql import _write_sine_wav

_FEATURES_HEADER = (
    "track_id,track_name,track_artist,danceability,energy,key,loudness,mode,"
    "speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms"
)


def _make_session(storage_dir: Path, resolver=None) -> Session:
    executor = QueryExecutor(
        EngineIndexFactory(media_resolver=resolver),
        FileStorageEngine(storage_dir),
    )
    return Session(SqlParser(), QueryPlanner(), executor, Catalog())


def _write_spotify_fixture(folder: Path) -> tuple[Path, Path, Path]:
    features = folder / "features.csv"
    lyrics = folder / "lyrics.csv"
    images = folder / "images"
    images.mkdir()
    (images / "t1.jpg").write_bytes(b"fake")
    zeros = ",".join("0.0" for _ in range(10))
    features.write_text(
        _FEATURES_HEADER + "\n"
        f"t1,Song One,Artist A,1.0,0.0,{zeros}\n"
        f"t2,Song Two,Artist B,0.0,1.0,{zeros}\n"
        f"t3,Song Three,Artist C,0.9,0.1,{zeros}\n",
        encoding="utf-8",
    )
    lyrics.write_text(
        "track_id,track_name,track_artist,lyrics\n"
        't1,Song One,Artist A,"visual search database"\n'
        't2,Song Two,Artist B,"audio retrieval engine"\n'
        't3,Song Three,Artist C,"storage engine design"\n'
        't9,Ghost,Artist X,"sin features no entra"\n',
        encoding="utf-8",
    )
    return features, lyrics, images


def _write_fma_fixture(folder: Path) -> tuple[Path, Path]:
    audio = folder / "audio"
    nested = audio / "000"
    nested.mkdir(parents=True)
    _write_sine_wav(nested / "000001.wav", freq=220.0)
    _write_sine_wav(nested / "000002.wav", freq=880.0)
    _write_sine_wav(nested / "000005.wav", freq=3520.0)
    tracks = folder / "tracks.csv"
    with open(tracks, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "album_title", "artist_name", "track_title", "track_genres"])
        writer.writerow(["1", "Album A", "Artist A", "Low Tone", "[{'genre_id': '1', 'genre_title': 'Rock'}]"])
        writer.writerow(["2", "Album B", "Artist B", "Mid Tone", "[{'genre_id': '2', 'genre_title': 'Jazz'}]"])
        writer.writerow(["5", "Album C", "Artist C", "High Tone", "[{'genre_id': '3', 'genre_title': 'Electronic'}]"])
        writer.writerow(["9", "Album D", "Artist D", "Missing", "[{'genre_id': '4', 'genre_title': 'Pop'}]"])
    return tracks, audio


def test_spotify_loader_joins_lyrics_features_and_images(tmp_path: Path) -> None:
    features, lyrics, images = _write_spotify_fixture(tmp_path)
    loader = SpotifyLoader(features, lyrics, images)

    rows = list(loader.rows())

    assert [row[1] for row in rows] == ["t1", "t2", "t3"]
    assert rows[0][5] == [1.0, 0.0] + [0.0] * 10
    assert rows[0][6] == "t1.jpg"
    assert rows[1][6] == ""


def test_fma_loader_parses_metadata_and_filters_missing_audio(tmp_path: Path) -> None:
    tracks, audio = _write_fma_fixture(tmp_path)
    loader = FMALoader(tracks, audio)

    rows = list(loader.rows())

    assert [row[1] for row in rows] == ["1", "2", "5"]
    assert rows[0][2] == "Low Tone"
    assert rows[0][3] == "Artist A"
    assert rows[0][4] == "Rock"
    assert rows[0][5] == "000001.wav"


def test_importer_spotify_end_to_end(tmp_path: Path) -> None:
    features, lyrics, images = _write_spotify_fixture(tmp_path)
    session = _make_session(tmp_path / "storage")
    loader = SpotifyLoader(features, lyrics, images)

    report = DatasetImporter(session, batch_size=2).run(loader)

    assert report.rows_inserted == 3
    assert report.batches == 2
    assert report.indexes == ["id:hash", "lyrics:inverted", "feat:knn"]

    match = session.execute('SELECT id FROM tracks WHERE MATCH(lyrics, "visual search", 2)')
    assert match.rows == [(1,)]

    query = "[" + ", ".join(["1.0", "0.0"] + ["0.0"] * 10) + "]"
    knn = session.execute(f"SELECT * FROM tracks WHERE KNN(feat, {query}, 2)")
    hits = [record for (record,) in knn.rows]
    assert [key for key, _score in hits] == ["1", "3"]


def test_importer_respects_limit(tmp_path: Path) -> None:
    features, lyrics, images = _write_spotify_fixture(tmp_path)
    session = _make_session(tmp_path / "storage")

    report = DatasetImporter(session).run(SpotifyLoader(features, lyrics, images), limit=2)

    assert report.rows_inserted == 2
    result = session.execute("SELECT id FROM tracks WHERE id = 2")
    assert result.rows == [(2,)]


def test_importer_fma_end_to_end_with_audio_knn(tmp_path: Path) -> None:
    from multimedia.codebook.kmeans_codebook import KMeansCodebook
    from multimedia.extractors.mfcc_extractor import MFCCExtractor
    from multimedia.resolver import PipelineMediaResolver

    tracks, audio = _write_fma_fixture(tmp_path)
    _write_sine_wav(audio / "query.wav", freq=230.0)
    resolver = PipelineMediaResolver(MFCCExtractor(), KMeansCodebook(k=4), audio)
    session = _make_session(tmp_path / "storage", resolver)

    report = DatasetImporter(session).run(FMALoader(tracks, audio))

    assert report.rows_inserted == 3

    result = session.execute('SELECT * FROM fma_tracks WHERE KNN(audio, "query.wav", 2)')
    hits = [record for (record,) in result.rows]
    assert hits[0][0] == "000001.wav"
    assert result.index_type == "knn"


def test_importer_sanitizes_multiline_and_quoted_text(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    lyrics = tmp_path / "lyrics.csv"
    zeros = ",".join("0.0" for _ in range(11))
    features.write_text(
        _FEATURES_HEADER + "\n" + f"t1,Song,Artist,1.0,{zeros}\n",
        encoding="utf-8",
    )
    lyrics.write_text(
        "track_id,track_name,track_artist,lyrics\n"
        't1,Song,Artist,"linea uno\ndice ""hola"" y sigue"\n',
        encoding="utf-8",
    )
    session = _make_session(tmp_path / "storage")

    report = DatasetImporter(session).run(SpotifyLoader(features, lyrics))

    assert report.rows_inserted == 1
    result = session.execute("SELECT lyrics FROM tracks WHERE id = 1")
    assert result.rows == [("linea uno dice 'hola' y sigue",)]


# Corpus mínimo con frecuencias controladas para probar la poda
class _TinyTextLoader(DatasetLoader):

    def table_name(self) -> str:
        return "docs"

    def columns(self) -> list[ColumnSpec]:
        return [ColumnSpec(name="id", type="INT"), ColumnSpec(name="body", type="TEXT")]

    def indexes(self) -> list[IndexInfo]:
        return [IndexInfo(column="body", index_type="inverted", options={"vocabulary": 2})]

    def rows(self) -> Iterator[tuple]:
        yield (1, "alpha alpha beta")
        yield (2, "alpha beta")
        yield (3, "gamma delta gamma")


class _RecordingSession:

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str):
        self.statements.append(sql)


def test_importer_serializes_index_options_as_with_clause() -> None:
    session = _RecordingSession()

    DatasetImporter(session).run(_TinyTextLoader())

    assert (
        "CREATE INDEX ON docs (body) USING inverted WITH (vocabulary = 2)"
        in session.statements
    )


def test_importer_vocabulary_option_prunes_rare_terms(tmp_path: Path) -> None:
    session = _make_session(tmp_path / "storage")

    report = DatasetImporter(session).run(_TinyTextLoader())

    assert report.rows_inserted == 3
    frequent = session.execute('SELECT id FROM docs WHERE MATCH(body, "alpha", 3)')
    rare = session.execute('SELECT id FROM docs WHERE MATCH(body, "delta", 3)')
    assert {row[0] for row in frequent.rows} == {1, 2}
    assert rare.rows == []


def test_spotify_loader_declares_lyrics_vocabulary() -> None:
    loader = SpotifyLoader("features.csv", "lyrics.csv")

    inverted = [ix for ix in loader.indexes() if ix.index_type == "inverted"]

    assert len(inverted) == 1
    assert inverted[0].options == {"vocabulary": 8000}
